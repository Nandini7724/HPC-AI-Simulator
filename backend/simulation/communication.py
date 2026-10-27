"""
communication.py
-----------------
COMMUNICATION OVERHEAD

Applies to any strategy that splits work across more than one device
(data / model / hybrid / distributed parallelism).

    Communication Time = Data Exchanged / Network Bandwidth
                          + Latency x Number of Communication Events

Data exchanged per synchronization event is modeled as a ring all-reduce
of the full model (weights or gradients), which moves roughly 2x the
model size across the network per event.

NOTE (documented simplification): a real system synchronizes gradients on
every mini-batch, but overlaps that transfer with the next batch's
computation and often compresses/buckets gradients. Modeling a literal,
un-overlapped all-reduce on every one of possibly millions of mini-batches
would make communication numerically dominate everything regardless of
the scenario, which is not representative and not useful for teaching.
As a documented simplification we model ONE amortized synchronization
event per epoch (representing the accumulated, overlapped communication
cost for that epoch). This keeps the event count tractable while still
responding correctly to model size, device count, bandwidth and latency.
"""
from dataclasses import dataclass
from . import config


@dataclass
class CommunicationResult:
    enabled: bool
    devices: int
    sync_events: float
    data_per_event_gb: float
    communication_time_sec: float


def evaluate_communication(strategy: str, devices: int, epochs: float,
                            model_parameters: int,
                            network_bandwidth_gbps: float = config.DEFAULT_NETWORK_BANDWIDTH_GBPS,
                            latency_ms: float = config.DEFAULT_LATENCY_MS) -> CommunicationResult:
    multi_device_strategies = {"data_parallel", "model_parallel",
                                "hybrid_parallel", "distributed"}
    if strategy not in multi_device_strategies or devices <= 1:
        return CommunicationResult(enabled=False, devices=max(devices, 1),
                                    sync_events=0.0, data_per_event_gb=0.0,
                                    communication_time_sec=0.0)

    # One amortized synchronization event per epoch (see module docstring).
    sync_events = max(epochs, 1.0)

    data_per_event_gb = (model_parameters * config.BYTES_PER_PARAM
                          * config.ALLREDUCE_FACTOR) / 1e9

    network_bandwidth_GBps = max(network_bandwidth_gbps / 8.0, 1e-6)  # Gbps -> GB/s

    transfer_time = sync_events * (data_per_event_gb / network_bandwidth_GBps)
    latency_time = sync_events * (latency_ms / 1000.0)
    # More devices in the group -> more peers to synchronize with each event.
    latency_time *= (1.0 + 0.15 * (devices - 1))

    communication_time_sec = transfer_time + latency_time

    return CommunicationResult(
        enabled=True,
        devices=devices,
        sync_events=sync_events,
        data_per_event_gb=data_per_event_gb,
        communication_time_sec=communication_time_sec,
    )
