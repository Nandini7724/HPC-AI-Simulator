"""
workload.py
-----------
WORKLOAD ANALYZER

Converts the user's AI workload configuration (dataset, model, batch size,
epochs) into:
  - an estimated number of training samples
  - an estimated total number of floating point operations (FLOPs)
  - an estimated memory footprint

This does NOT run any real training. It is a mathematical estimate,
clearly labeled as such wherever it is displayed.
"""
from dataclasses import dataclass
from . import config


@dataclass
class WorkloadProfile:
    total_samples: float
    batches_per_epoch: float
    total_batches: float
    total_operations: float          # total FLOPs across all epochs
    memory_required_gb: float        # estimated working-set memory


def analyze_workload(dataset_size_gb: float, model_parameters: int,
                      batch_size: int, epochs: int,
                      avg_sample_size_mb: float = config.DEFAULT_AVG_SAMPLE_SIZE_MB
                      ) -> WorkloadProfile:
    """
    Core workload model:

        Total Samples      = Dataset Size / Average Sample Size
        Batches per Epoch  = Total Samples / Batch Size
        Total Batches      = Batches per Epoch x Epochs
        Ops per Sample     = Model Parameters x FLOPS_PER_PARAM_TRAIN
        Total Operations   = Ops per Sample x Total Samples x Epochs
    """
    dataset_size_mb = dataset_size_gb * 1024.0
    total_samples = max(dataset_size_mb / avg_sample_size_mb, 1.0)

    batches_per_epoch = max(total_samples / batch_size, 1.0)
    total_batches = batches_per_epoch * epochs

    ops_per_sample = model_parameters * config.FLOPS_PER_PARAM_TRAIN
    total_operations = ops_per_sample * total_samples * epochs

    # Working-set memory: model weights + gradients + optimizer state,
    # plus the memory needed to hold one batch of activations/inputs.
    model_state_gb = (model_parameters * config.BYTES_PER_PARAM
                       * config.MEMORY_STATE_MULTIPLIER) / 1e9
    batch_gb = (batch_size * avg_sample_size_mb) / 1024.0
    memory_required_gb = model_state_gb + batch_gb

    return WorkloadProfile(
        total_samples=total_samples,
        batches_per_epoch=batches_per_epoch,
        total_batches=total_batches,
        total_operations=total_operations,
        memory_required_gb=memory_required_gb,
    )
