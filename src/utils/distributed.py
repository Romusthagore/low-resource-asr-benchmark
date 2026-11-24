"""Utilities for single-GPU and distributed training."""

import math
import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with NCCL requires CUDA.")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        dist.init_process_group(backend="nccl")

        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        rank = 0
        local_rank = 0

    return {
        "distributed": distributed,
        "device": device,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def wrap_model(model, context):
    if context["distributed"]:
        return DistributedDataParallel(
            model,
            device_ids=[context["local_rank"]],
            find_unused_parameters=True,
        )

    return model


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module

    return model


def is_main_process(context):
    return context["rank"] == 0


def barrier(context):
    if context["distributed"]:
        dist.barrier()


def cleanup_distributed(context):
    if context["distributed"] and dist.is_initialized():
        dist.destroy_process_group()


def distributed_epoch_indices(
    dataset_size,
    epoch,
    seed,
    rank,
    world_size,
):
    """Build deterministic, equal-sized per-rank training indices."""

    if dataset_size < 1:
        raise ValueError("Training dataset must not be empty")

    indices = list(range(dataset_size))
    rng = random.Random(seed + epoch)
    rng.shuffle(indices)

    if world_size == 1:
        return indices

    samples_per_rank = math.ceil(dataset_size / world_size)
    total_size = samples_per_rank * world_size

    if total_size > dataset_size:
        repeats = total_size - dataset_size
        indices.extend(
            (indices * math.ceil(repeats / dataset_size))[:repeats]
        )

    return indices[rank:total_size:world_size]


def gather_rank_states(local_state, context):
    """Collect one serializable state per distributed process."""

    if not context["distributed"]:
        return [local_state]

    gathered = [None] * context["world_size"]
    dist.all_gather_object(gathered, local_state)

    return gathered
