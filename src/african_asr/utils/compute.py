"""Compute measurement utilities for ASR training."""

import json
import time
from pathlib import Path

import torch


class ComputeTracker:
    """Track training time and GPU compute statistics."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.device_count = 0
        self.device_names = []

    def start(self):
        """Start compute measurement."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            self.device_count = torch.cuda.device_count()
            self.device_names = [
                torch.cuda.get_device_name(i)
                for i in range(self.device_count)
            ]
        else:
            self.device_count = 0
            self.device_names = []

        self.start_time = time.perf_counter()

    def stop(self):
        """Stop compute measurement and return results."""
        if self.start_time is None:
            raise RuntimeError("ComputeTracker.start() must be called first.")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.end_time = time.perf_counter()

        elapsed_seconds = self.end_time - self.start_time
        elapsed_hours = elapsed_seconds / 3600.0

        peak_memory_gb = 0.0

        if torch.cuda.is_available():
            peak_memory_gb = max(
                torch.cuda.max_memory_allocated(i)
                for i in range(self.device_count)
            ) / (1024 ** 3)

        return {
            "training_time_seconds": elapsed_seconds,
            "training_time_hours": elapsed_hours,
            "gpu_count": self.device_count,
            "gpu_names": self.device_names,
            "gpu_hours": elapsed_hours * self.device_count,
            "peak_gpu_memory_gb": peak_memory_gb,
        }

    def save(self, results, output_path):
        """Save compute measurements as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
