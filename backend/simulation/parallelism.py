"""
parallelism.py
---------------
PARALLELIZATION ENGINE

Resolves a chosen strategy into concrete resource usage:

    sequential        -> 1 CPU core, no GPU, no communication
    cpu_multicore     -> all requested CPU cores, no GPU
    data_parallel     -> CPU + GPU(s), dataset split across devices
    model_parallel    -> CPU + GPU(s), model layers split across devices
    hybrid_parallel   -> CPU + GPU(s), data AND model split
    distributed       -> CPU + GPU(s) treated as a multi-node cluster

The "devices" count returned is what communication.py and the efficiency
calculation use as the number of parallel processing units.
"""
from dataclasses import dataclass
from . import config


@dataclass
class ParallelPlan:
    strategy: str
    effective_cores: int
    effective_gpus: int
    devices: int              # processing units used for efficiency/speedup
    communication_applies: bool
    description: str


STRATEGY_DESCRIPTIONS = {
    "sequential": "Single core, no parallelism. Used as the performance baseline.",
    "cpu_multicore": "Workload is split across CPU cores on a single machine.",
    "data_parallel": "The dataset is split across devices; each device holds a full model copy.",
    "model_parallel": "Different layers/parts of the model are placed on different devices.",
    "hybrid_parallel": "Combines data parallelism and model parallelism across devices.",
    "distributed": "Data/model parallelism extended across multiple networked nodes.",
}


def resolve_strategy(strategy: str, cpu_cores: int, gpus: int) -> ParallelPlan:
    strategy = strategy if strategy in config.PARALLEL_STRATEGIES else "sequential"
    cpu_cores = max(int(cpu_cores), 1)
    gpus = max(int(gpus), 0)

    if strategy == "sequential":
        eff_cores, eff_gpus = 1, 0
    elif strategy == "cpu_multicore":
        eff_cores, eff_gpus = cpu_cores, 0
    else:
        # data_parallel / model_parallel / hybrid_parallel / distributed
        eff_cores, eff_gpus = cpu_cores, gpus

    if eff_gpus > 0:
        devices = eff_gpus
    else:
        devices = eff_cores

    communication_applies = strategy in (
        "data_parallel", "model_parallel", "hybrid_parallel", "distributed"
    ) and devices > 1

    return ParallelPlan(
        strategy=strategy,
        effective_cores=eff_cores,
        effective_gpus=eff_gpus,
        devices=devices,
        communication_applies=communication_applies,
        description=STRATEGY_DESCRIPTIONS.get(strategy, ""),
    )
