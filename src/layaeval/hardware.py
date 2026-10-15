"""Hardware infrastructure and accelerator detection for LayaEval."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """Detected hardware infrastructure details."""

    device_type: str  # "cuda", "mps", "cpu"
    name: str
    description: str
    is_gpu: bool
    cores: int
    memory_total_gb: float | None = None
    cuda_device_index: int | None = None

    def __str__(self) -> str:
        return self.description


_DETECTED_HARDWARE: HardwareInfo | None = None


def detect_hardware() -> HardwareInfo:
    """Detect available compute infrastructure (CUDA GPU, Apple MPS, or CPU).

    Caches the result after first detection for sub-millisecond lookups.
    """
    global _DETECTED_HARDWARE
    if _DETECTED_HARDWARE is not None:
        return _DETECTED_HARDWARE

    cores = os.cpu_count() or 1

    # 1. Check PyTorch if available
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            mem_gb = round(props.total_memory / (1024**3), 1)
            desc = f"GPU: {gpu_name} (CUDA, {mem_gb} GB VRAM, Device {idx})"
            info = HardwareInfo(
                device_type="cuda",
                name=gpu_name,
                description=desc,
                is_gpu=True,
                cores=cores,
                memory_total_gb=mem_gb,
                cuda_device_index=idx,
            )
            _DETECTED_HARDWARE = info
            return info

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            desc = f"GPU: Apple Silicon Metal/MPS ({cores} cores)"
            info = HardwareInfo(
                device_type="mps",
                name="Apple Silicon (MPS)",
                description=desc,
                is_gpu=True,
                cores=cores,
            )
            _DETECTED_HARDWARE = info
            return info

        desc = f"CPU: {platform.machine()} ({cores} cores)"
        info = HardwareInfo(
            device_type="cpu",
            name=f"CPU ({platform.machine()})",
            description=desc,
            is_gpu=False,
            cores=cores,
        )
        _DETECTED_HARDWARE = info
        return info

    except ImportError:
        pass
    except Exception:
        pass

    # 2. Check nvidia-smi if torch not imported/installed
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
            if out:
                first_line = out.split("\n")[0]
                parts = [p.strip() for p in first_line.split(",")]
                gpu_name = parts[0]
                mem_gb = round(float(parts[1]) / 1024.0, 1) if len(parts) > 1 else None
                mem_str = f", {mem_gb} GB VRAM" if mem_gb else ""
                desc = f"GPU: {gpu_name} (CUDA{mem_str})"
                info = HardwareInfo(
                    device_type="cuda",
                    name=gpu_name,
                    description=desc,
                    is_gpu=True,
                    cores=cores,
                    memory_total_gb=mem_gb,
                    cuda_device_index=0,
                )
                _DETECTED_HARDWARE = info
                return info
        except Exception:
            pass

    # 3. Apple Silicon MPS on macOS
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        desc = f"GPU: Apple Silicon Metal/MPS ({cores} cores)"
        info = HardwareInfo(
            device_type="mps",
            name="Apple Silicon (MPS)",
            description=desc,
            is_gpu=True,
            cores=cores,
        )
        _DETECTED_HARDWARE = info
        return info

    # 4. Standard CPU fallback
    arch = platform.machine() or "x86_64"
    desc = f"CPU: {arch} ({cores} cores)"
    info = HardwareInfo(
        device_type="cpu",
        name=f"CPU ({arch})",
        description=desc,
        is_gpu=False,
        cores=cores,
    )
    _DETECTED_HARDWARE = info
    return info
