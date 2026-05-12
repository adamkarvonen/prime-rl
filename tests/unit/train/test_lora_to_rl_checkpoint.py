from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from prime_rl.tools.lora_to_rl_checkpoint import convert_adapter, convert_adapter_to_checkpoint, load_rl_config


def _write_rl_config(
    path: Path,
    output_dir: Path,
    rank: int = 2,
    alpha: float = 4.0,
    max_concurrent_runs: int = 2,
) -> None:
    single_run_ckpt = "skip_optimizer = true" if max_concurrent_runs == 1 else ""
    path.write_text(
        f"""
output_dir = "{output_dir.as_posix()}"
max_steps = 1
seq_len = 128

[ckpt]
resume_step = 0

[trainer.ckpt]
{single_run_ckpt}

[model]
name = "Base/Model"

[trainer]
max_concurrent_runs = {max_concurrent_runs}

[trainer.model.lora]
rank = {rank}
alpha = {alpha}
dropout = 0.1
target_modules = ["q_proj"]

[orchestrator]
batch_size = 2
rollouts_per_example = 1

[orchestrator.model.lora]
name = "warm"

[orchestrator.train.sampling]
temperature = 1.0
max_completion_tokens = 16

[[orchestrator.train.env]]
id = "reverse-text"
"""
    )


def _write_adapter(adapter_dir: Path, rank: int = 2, alpha: float = 4.0) -> None:
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        f"""
{{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "base_model_name_or_path": "Base/Model",
  "r": {rank},
  "lora_alpha": {alpha},
  "lora_dropout": 0.1,
  "bias": "none",
  "target_modules": ["q_proj"],
  "modules_to_save": null
}}
"""
    )
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor(
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]]
            ),
        },
        adapter_dir / "adapter_model.safetensors",
    )


def test_convert_adapter_to_checkpoint_writes_multi_run_prime_rl_step_zero(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "rl.toml"
    _write_adapter(adapter_dir)
    _write_rl_config(config_path, output_dir)

    step_dir = convert_adapter_to_checkpoint(
        adapter_dir=adapter_dir,
        rl_config_path=config_path,
        output_dir=output_dir,
        run_id="run_default",
        step=0,
        trainer_rank_count=1,
        overwrite=False,
    )

    assert step_dir == output_dir / "run_default" / "checkpoints" / "step_0"
    assert (step_dir / "STABLE").exists()
    trainer_state = torch.load(step_dir / "trainer" / "rank_0.pt", weights_only=False)
    assert set(trainer_state) == {"model", "progress"}
    assert trainer_state["progress"]["step"] == 0
    assert set(trainer_state["model"]) == {
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    }

    weight_state = load_file(step_dir / "weight" / "adapter_model.safetensors", device="cpu")
    assert (step_dir / "weight" / "STABLE").exists()
    assert set(weight_state) == set(trainer_state["model"])
    assert torch.equal(
        weight_state["model.layers.0.self_attn.q_proj.lora_A.weight"],
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )

    orchestrator_state = torch.load(step_dir / "orchestrator" / "progress.pt", weights_only=False)
    assert orchestrator_state["progress"].step == 0
    for filename in ("easy_examples.jsonl", "hard_examples.jsonl", "rollout_buffer.jsonl"):
        assert (step_dir / "orchestrator" / "buffer" / filename).read_text() == ""


def test_convert_adapter_to_checkpoint_writes_single_run_prime_rl_step_zero(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "rl.toml"
    _write_adapter(adapter_dir)
    _write_rl_config(config_path, output_dir, max_concurrent_runs=1)

    step_dir = convert_adapter_to_checkpoint(
        adapter_dir=adapter_dir,
        rl_config_path=config_path,
        output_dir=output_dir,
        run_id="run_default",
        step=0,
        trainer_rank_count=1,
        overwrite=False,
    )

    assert step_dir == output_dir / "run_default" / "checkpoints" / "step_0"
    trainer_dir = output_dir / "checkpoints" / "step_0" / "trainer"
    assert (output_dir / "checkpoints" / "step_0" / "STABLE").exists()
    assert (trainer_dir / ".metadata").exists()
    assert any(path.name.endswith(".distcp") for path in trainer_dir.iterdir())

    assert (step_dir / "STABLE").exists()
    weight_state = load_file(step_dir / "weight" / "adapter_model.safetensors", device="cpu")
    assert set(weight_state) == {
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    }

    metadata = (step_dir / "conversion_metadata.json").read_text()
    assert '"max_concurrent_runs": 1' in metadata
    assert f'"trainer_checkpoint_dir": "{trainer_dir.as_posix()}"' in metadata


def test_convert_adapter_to_checkpoint_writes_each_trainer_rank(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "rl.toml"
    _write_adapter(adapter_dir)
    _write_rl_config(config_path, output_dir)

    step_dir = convert_adapter_to_checkpoint(
        adapter_dir=adapter_dir,
        rl_config_path=config_path,
        output_dir=output_dir,
        run_id="run_default",
        step=0,
        trainer_rank_count=2,
        overwrite=False,
    )

    rank_0 = torch.load(step_dir / "trainer" / "rank_0.pt", weights_only=False)
    rank_1 = torch.load(step_dir / "trainer" / "rank_1.pt", weights_only=False)
    assert rank_0["progress"] == rank_1["progress"]
    assert set(rank_0["model"]) == set(rank_1["model"])
    for key in rank_0["model"]:
        assert torch.equal(rank_0["model"][key], rank_1["model"][key])


def test_convert_adapter_rejects_rank_mismatch(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "rl.toml"
    _write_adapter(adapter_dir, rank=4)
    _write_rl_config(config_path, output_dir, rank=2)
    config = load_rl_config(config_path, output_dir)

    from prime_rl.tools.lora_to_rl_checkpoint import settings_from_rl_config

    settings = settings_from_rl_config(config, step=0)
    with pytest.raises(AssertionError, match="rank"):
        convert_adapter(adapter_dir, settings)


def test_convert_adapter_stacks_moe_expert_slices(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        """
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "base_model_name_or_path": "Base/Model",
  "r": 1,
  "lora_alpha": 2.0,
  "lora_dropout": 0.0,
  "bias": "none",
  "target_modules": ["experts"],
  "modules_to_save": null
}
"""
    )
    save_file(
        {
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "base_model.model.model.layers.0.mlp.experts.1.gate_proj.lora_A.weight": torch.tensor([[3.0, 4.0]]),
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.tensor([[5.0], [6.0]]),
            "base_model.model.model.layers.0.mlp.experts.1.gate_proj.lora_B.weight": torch.tensor([[7.0], [8.0]]),
        },
        adapter_dir / "adapter_model.safetensors",
    )

    from prime_rl.tools.lora_to_rl_checkpoint import WarmStartSettings

    converted = convert_adapter(
        adapter_dir,
        WarmStartSettings(
            model_name="Base/Model",
            rank=1,
            alpha=2.0,
            dropout=0.0,
            target_modules=["experts"],
            torch_dtype=torch.float32,
            step=0,
            max_concurrent_runs=2,
        ),
    )

    assert torch.equal(
        converted.trainer_state_dict["model.layers.0.mlp.experts.w1_lora_A.weight"],
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
    )
    assert "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight" in converted.weight_state_dict
