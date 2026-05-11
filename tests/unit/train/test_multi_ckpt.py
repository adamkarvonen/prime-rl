from pathlib import Path
from typing import Any, Generator, cast

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, distribute_tensor

from prime_rl.trainer.multi_ckpt import MultiCheckpointManager, RunState
from prime_rl.trainer.runs import Progress


def test_load_state_dict_copies_plain_tensor() -> None:
    target = torch.zeros((2, 3), dtype=torch.bfloat16)
    checkpoint_value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    run_state = RunState({"adapter": target}, None, None, Progress())

    run_state.load_state_dict({"model": {"adapter": checkpoint_value}})

    assert torch.equal(target, checkpoint_value.to(dtype=torch.bfloat16))


@pytest.fixture(scope="module", autouse=True)
def init_process_group(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    if dist.is_initialized():
        yield
        return

    rendezvous_path = Path(tmp_path_factory.mktemp("process_group")) / "init"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=0,
        world_size=1,
    )
    yield
    dist.destroy_process_group()


def test_load_state_dict_distributes_plain_tensor_into_dtensor() -> None:
    device_mesh = init_device_mesh("cpu", (1,))
    target_tensor = torch.zeros((2, 3), dtype=torch.bfloat16)
    target = distribute_tensor(target_tensor, device_mesh=device_mesh, placements=[Replicate()])
    checkpoint_value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    run_state = RunState({"adapter": target}, None, None, Progress())

    run_state.load_state_dict({"model": {"adapter": checkpoint_value}})

    assert torch.equal(target.full_tensor(), checkpoint_value.to(dtype=torch.bfloat16))


def test_load_state_dict_fails_on_shape_mismatch() -> None:
    run_state = RunState({"adapter": torch.zeros((2, 3))}, None, None, Progress())

    with pytest.raises(AssertionError, match="Checkpoint tensor shape mismatch"):
        run_state.load_state_dict({"model": {"adapter": torch.zeros((3, 2))}})


def test_multi_checkpoint_save_accepts_trainer_checkpoint_signature() -> None:
    manager = MultiCheckpointManager.__new__(MultiCheckpointManager)
    manager.multi_run_manager = cast(Any, type("EmptyRunManager", (), {"used_idxs": []})())

    manager.save(100, torch.nn.Linear(1, 1), [cast(Any, object())], cast(Any, object()), Progress())
