"""Small, bounded local environment inspection with no network requests."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def inspect_environment(storage: Path) -> dict:
    storage = storage.resolve(strict=True)
    if not storage.is_dir():
        raise OSError("Storage path must be an existing directory")
    tools = {name: shutil.which(name) is not None for name in ("git", "ffmpeg", "nvidia-smi")}
    gpu = {"status": "unavailable", "devices": []}
    gpu_command = shutil.which("nvidia-smi")
    if gpu_command:
        try:
            result = subprocess.run(
                [gpu_command, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.rsplit(",", 2)]
                    if len(parts) == 3:
                        gpu["devices"].append({"name": parts[0], "memory_mib": parts[1], "driver": parts[2]})
                gpu["status"] = "detected" if gpu["devices"] else "no_devices"
            else:
                gpu["status"] = "query_failed"
        except (OSError, subprocess.TimeoutExpired, UnicodeError):
            gpu["status"] = "query_failed"
    return {
        "report_version": 1,
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 11),
        "system": platform.system(),
        "machine": platform.machine(),
        "storage_free_gib": round(shutil.disk_usage(storage).free / 1024**3, 2),
        "tools": tools,
        "gpu": gpu,
        "runtime_ready": False,
        "limitations": [
            "Preparation utilities only; capture, input, training and GUI are not implemented.",
            "GPU detection does not prove PyTorch/CUDA compatibility or game performance.",
            "No game was opened and no control input was sent.",
        ],
    }
