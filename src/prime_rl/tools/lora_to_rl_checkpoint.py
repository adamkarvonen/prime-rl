"""Convert a PEFT LoRA adapter into a PrimeRL warm-start checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import tomllib
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from safetensors.torch import load_file, save_file

from prime_rl.configs.rl import RLConfig
from prime_rl.orchestrator.ckpt import Progress as OrchestratorProgress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.runs import Progress as TrainerProgress

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_MODEL_NAME = "adapter_model.safetensors"
PEFT_PREFIX = "base_model.model."

_MOE_LORA_KEY_RE = re.compile(
    r"(?P<prefix>.*\.experts)\.(?P<eid>\d+)\.(?P<proj>gate_proj|down_proj|up_proj)\.(?P<ab>lora_[AB])(?:\.(?:default|\d+))?(?:\.weight)?$"
)


@dataclass(frozen=True)
class WarmStartSettings:
    model_name: str
    rank: int
    alpha: float
    dropout: float
    target_modules: list[str]
    torch_dtype: torch.dtype
    step: int
    max_concurrent_runs: int


@dataclass(frozen=True)
class ConvertedAdapter:
    trainer_state_dict: dict[str, torch.Tensor]
    weight_state_dict: dict[str, torch.Tensor]
    adapter_config: dict[str, Any]


def load_rl_config(config_path: Path, output_dir: Path | None) -> RLConfig:
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)
    if output_dir is not None:
        config_dict["output_dir"] = output_dir
    return RLConfig(**config_dict)


def settings_from_rl_config(config: RLConfig, step: int) -> WarmStartSettings:
    trainer_lora = config.trainer.model.lora
    assert trainer_lora is not None, "Trainer config must enable model.lora."
    assert config.trainer.max_concurrent_runs >= 1, (
        f"trainer.max_concurrent_runs must be >= 1, got {config.trainer.max_concurrent_runs}."
    )
    assert config.trainer.ckpt is not None, "Trainer checkpoint config is required."
    assert config.orchestrator.ckpt is not None, "Orchestrator checkpoint config is required."
    assert config.trainer.ckpt.resume_step == step, (
        f"trainer.ckpt.resume_step must equal converter step {step}, got {config.trainer.ckpt.resume_step}."
    )
    assert config.orchestrator.ckpt.resume_step == step, (
        f"orchestrator.ckpt.resume_step must equal converter step {step}, got {config.orchestrator.ckpt.resume_step}."
    )
    if config.trainer.max_concurrent_runs == 1:
        assert step == 0, "Single-run LoRA warm-start checkpoints only support step 0."
        assert config.trainer.ckpt.skip_optimizer, (
            "Single-run LoRA warm-start checkpoints intentionally do not include optimizer state. "
            "Set trainer.ckpt.skip_optimizer = true."
        )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    return WarmStartSettings(
        model_name=config.trainer.model.name,
        rank=trainer_lora.rank,
        alpha=trainer_lora.alpha,
        dropout=trainer_lora.dropout,
        target_modules=trainer_lora.target_modules,
        torch_dtype=dtype_map[config.trainer.model.optimization_dtype],
        step=step,
        max_concurrent_runs=config.trainer.max_concurrent_runs,
    )


def _load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    with open(adapter_dir / ADAPTER_CONFIG_NAME) as f:
        adapter_config = json.load(f)
    return adapter_config


def _validate_adapter_config(adapter_config: dict[str, Any], settings: WarmStartSettings) -> None:
    assert adapter_config["peft_type"] == "LORA", f"Expected LoRA adapter, got {adapter_config['peft_type']}."
    assert int(adapter_config["r"]) == settings.rank, (
        f"Adapter rank {adapter_config['r']} does not match trainer rank {settings.rank}."
    )
    assert float(adapter_config["lora_alpha"]) == settings.alpha, (
        f"Adapter alpha {adapter_config['lora_alpha']} does not match trainer alpha {settings.alpha}."
    )
    assert float(adapter_config["lora_dropout"]) == settings.dropout, (
        f"Adapter dropout {adapter_config['lora_dropout']} does not match trainer dropout {settings.dropout}."
    )
    assert adapter_config["modules_to_save"] in (None, []), (
        "Adapters with modules_to_save are not supported by this converter."
    )
    assert adapter_config["base_model_name_or_path"] == settings.model_name, (
        "Adapter base_model_name_or_path does not match the RL trainer model. "
        f"adapter={adapter_config['base_model_name_or_path']}, trainer={settings.model_name}"
    )
    adapter_targets = adapter_config["target_modules"]
    assert isinstance(adapter_targets, list), f"Adapter target_modules must be a list, got {type(adapter_targets)}."
    assert set(adapter_targets) == set(settings.target_modules), (
        f"Adapter target_modules {sorted(adapter_targets)} do not match trainer target_modules "
        f"{sorted(settings.target_modules)}."
    )


def _strip_peft_prefix(key: str) -> str:
    if key.startswith(PEFT_PREFIX):
        return key[len(PEFT_PREFIX) :]
    return key


def _normalize_dense_lora_key(key: str) -> str:
    key = _strip_peft_prefix(key)
    key = re.sub(r"\.(lora_[AB])\.(default|\d+)\.weight$", r".\1.weight", key)
    key = re.sub(r"\.(lora_[AB])\.(default|\d+)$", r".\1.weight", key)
    if key.endswith(".lora_A") or key.endswith(".lora_B"):
        key = f"{key}.weight"
    assert key.endswith(".weight"), f"LoRA key did not normalize to a weight tensor key: {key}"
    return key


def _parse_moe_lora_key(key: str) -> tuple[str, int, str] | None:
    key = _strip_peft_prefix(key)
    match = _MOE_LORA_KEY_RE.fullmatch(key)
    if match is None:
        return None
    proj_map = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
    trainer_key = f"{match.group('prefix')}.{proj_map[match.group('proj')]}_{match.group('ab')}.weight"
    weight_key = (
        f"{match.group('prefix')}.{match.group('eid')}.{match.group('proj')}.{match.group('ab')}.weight"
    )
    return trainer_key, int(match.group("eid")), weight_key


def _minimal_adapter_config(adapter_config: dict[str, Any], settings: WarmStartSettings) -> dict[str, Any]:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": settings.model_name,
        "r": settings.rank,
        "lora_alpha": settings.alpha,
        "lora_dropout": settings.dropout,
        "bias": adapter_config["bias"],
        "target_modules": sorted(adapter_config["target_modules"]),
        "modules_to_save": None,
    }


def convert_adapter(adapter_dir: Path, settings: WarmStartSettings) -> ConvertedAdapter:
    adapter_config = _load_adapter_config(adapter_dir)
    _validate_adapter_config(adapter_config, settings)

    adapter_model_path = adapter_dir / ADAPTER_MODEL_NAME
    assert adapter_model_path.exists(), f"Adapter weights not found: {adapter_model_path}"
    raw_state_dict = load_file(adapter_model_path, device="cpu")

    trainer_state_dict: dict[str, torch.Tensor] = {}
    weight_state_dict: dict[str, torch.Tensor] = {}
    moe_parts: dict[str, dict[int, torch.Tensor]] = {}

    for key, tensor in raw_state_dict.items():
        if "lora_A" not in key and "lora_B" not in key:
            raise ValueError(f"Unexpected non-LoRA tensor in adapter: {key}")
        moe_key = _parse_moe_lora_key(key)
        if moe_key is None:
            normalized_key = _normalize_dense_lora_key(key)
            trainer_state_dict[normalized_key] = tensor.detach().cpu().to(settings.torch_dtype)
            weight_state_dict[normalized_key] = tensor.detach().cpu().to(settings.torch_dtype)
        else:
            trainer_key, expert_id, weight_key = moe_key
            moe_parts.setdefault(trainer_key, {})[expert_id] = tensor.detach().cpu().to(settings.torch_dtype)
            weight_state_dict[weight_key] = tensor.detach().cpu().to(settings.torch_dtype)

    for trainer_key, parts in moe_parts.items():
        expert_ids = set(parts)
        assert expert_ids == set(range(len(parts))), f"Missing MoE expert slices for {trainer_key}: {sorted(expert_ids)}"
        trainer_state_dict[trainer_key] = torch.stack([parts[i] for i in range(len(parts))], dim=0)

    assert trainer_state_dict, "Converted adapter state dict is empty."
    assert weight_state_dict, "Converted adapter weight state dict is empty."

    return ConvertedAdapter(
        trainer_state_dict=trainer_state_dict,
        weight_state_dict=weight_state_dict,
        adapter_config=_minimal_adapter_config(adapter_config, settings),
    )


def write_prime_rl_checkpoint(
    converted: ConvertedAdapter,
    config: RLConfig,
    output_dir: Path,
    run_id: str,
    step: int,
    trainer_rank_count: int,
    overwrite: bool,
) -> Path:
    assert trainer_rank_count >= 1, f"trainer_rank_count must be >= 1, got {trainer_rank_count}."

    run_step_dir = output_dir / run_id / "checkpoints" / f"step_{step}"
    if run_step_dir.exists():
        assert overwrite, f"Checkpoint directory already exists: {run_step_dir}"
        shutil.rmtree(run_step_dir)

    orchestrator_dir = run_step_dir / "orchestrator"
    buffer_dir = orchestrator_dir / "buffer"
    weight_dir = run_step_dir / "weight"
    buffer_dir.mkdir(parents=True)
    weight_dir.mkdir(parents=True)

    trainer_ckpt_dir: Path
    if config.trainer.max_concurrent_runs == 1:
        trainer_step_dir = output_dir / "checkpoints" / f"step_{step}"
        if trainer_step_dir.exists():
            assert overwrite, f"Checkpoint directory already exists: {trainer_step_dir}"
            shutil.rmtree(trainer_step_dir)
        trainer_ckpt_dir = trainer_step_dir / "trainer"
        trainer_ckpt_dir.mkdir(parents=True)
        scheduler_state = _initial_scheduler_state(config)
        dcp_save(
            {
                "app": {
                    "model": converted.trainer_state_dict,
                    "optimizers": {},
                    "scheduler": scheduler_state,
                    "progress": asdict(TrainerProgress(step=step)),
                }
            },
            checkpoint_id=trainer_ckpt_dir,
            no_dist=True,
        )
        (trainer_step_dir / "STABLE").touch()
    else:
        trainer_ckpt_dir = run_step_dir / "trainer"
        trainer_ckpt_dir.mkdir(parents=True)
        trainer_state = {
            "model": converted.trainer_state_dict,
            "progress": asdict(TrainerProgress(step=step)),
        }
        for rank in range(trainer_rank_count):
            torch.save(trainer_state, trainer_ckpt_dir / f"rank_{rank}.pt")

    with open(orchestrator_dir / "progress.pt", "wb") as f:
        torch.save({"progress": OrchestratorProgress(step=step)}, f)
    for filename in ("easy_examples.jsonl", "hard_examples.jsonl", "rollout_buffer.jsonl"):
        (buffer_dir / filename).write_text("")

    save_file(converted.weight_state_dict, weight_dir / ADAPTER_MODEL_NAME, metadata={"format": "pt"})
    (weight_dir / ADAPTER_CONFIG_NAME).write_text(json.dumps(converted.adapter_config, indent=2) + "\n")
    (weight_dir / "STABLE").touch()

    metadata = {
        "format": "prime_rl_lora_warm_start",
        "run_id": run_id,
        "step": step,
        "max_concurrent_runs": config.trainer.max_concurrent_runs,
        "trainer_rank_count": trainer_rank_count,
        "trainer_checkpoint_dir": str(trainer_ckpt_dir),
        "num_tensors": len(converted.trainer_state_dict),
        "torch_dtype": str(next(iter(converted.trainer_state_dict.values())).dtype),
        "adapter_config": converted.adapter_config,
    }
    (run_step_dir / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_step_dir / "STABLE").touch()
    return run_step_dir


def _initial_scheduler_state(config: RLConfig) -> dict[str, Any]:
    dummy_param = torch.nn.Parameter(torch.zeros(()))
    dummy_optimizer = torch.optim.AdamW([dummy_param], lr=config.trainer.optim.lr)
    scheduler = setup_scheduler(
        dummy_optimizer,
        config.trainer.scheduler,
        config.trainer.max_steps,
        config.trainer.optim.lr,
    )
    return scheduler.state_dict()


def convert_adapter_to_checkpoint(
    adapter_dir: Path,
    rl_config_path: Path,
    output_dir: Path,
    run_id: str,
    step: int,
    trainer_rank_count: int,
    overwrite: bool,
) -> Path:
    config = load_rl_config(rl_config_path, output_dir)
    settings = settings_from_rl_config(config, step)
    converted = convert_adapter(adapter_dir, settings)
    return write_prime_rl_checkpoint(
        converted=converted,
        config=config,
        output_dir=output_dir,
        run_id=run_id,
        step=step,
        trainer_rank_count=trainer_rank_count,
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--rl-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--trainer-rank-count", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    step_dir = convert_adapter_to_checkpoint(
        adapter_dir=args.adapter_dir,
        rl_config_path=args.rl_config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        step=args.step,
        trainer_rank_count=args.trainer_rank_count,
        overwrite=args.overwrite,
    )
    print(step_dir)


if __name__ == "__main__":
    main()
