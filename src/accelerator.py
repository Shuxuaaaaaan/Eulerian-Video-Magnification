"""
Unified acceleration backend for CPU and CUDA computation.

Provides a transparent abstraction layer that switches between
NumPy (CPU) and CuPy (CUDA) backends based on user configuration.
"""

import os
import platform
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Global backend state
# ---------------------------------------------------------------------------

_backend = "cpu"  # "cpu" or "cuda"
_xp = np  # numpy or cupy module reference
_threads = 1


def init(accel: str = "cpu", threads: int = 1):
    """Initialize the acceleration backend.

    Args:
        accel: "cpu" or "cuda"
        threads: Number of worker processes for CPU mode (default 1 = serial)
    """
    global _backend, _xp, _threads
    _backend = accel.lower()
    _threads = max(1, threads)

    if _backend == "cuda":
        try:
            import cupy as cp
            _xp = cp
            # Warm-up: trigger lazy CUDA context initialization
            cp.array([0.0])
        except ImportError:
            print("[ERROR] CuPy is not installed. Install with: uv pip install cupy-cuda12x")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] CUDA initialization failed: {e}")
            sys.exit(1)
    else:
        _xp = np


def get_xp():
    """Return the current array module (numpy or cupy)."""
    return _xp


def get_backend() -> str:
    """Return the current backend name."""
    return _backend


def get_threads() -> int:
    """Return the configured number of worker processes."""
    return _threads


# ---------------------------------------------------------------------------
# Data transfer helpers
# ---------------------------------------------------------------------------


def to_device(arr):
    """Transfer a numpy array to the current device (no-op for CPU)."""
    if _backend == "cuda":
        return _xp.asarray(arr)
    return arr


def to_host(arr):
    """Transfer an array back to CPU numpy (no-op for CPU)."""
    if _backend == "cuda":
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Unified 2D convolution
# ---------------------------------------------------------------------------


def convolve2d(image, kernel):
    """2D convolution compatible with both backends.

    CPU mode  -> cv2.filter2D (fast, C++ optimized)
    CUDA mode -> cupyx.scipy.ndimage.convolve (GPU)
    """
    if _backend == "cuda":
        from cupyx.scipy.ndimage import convolve
        # convolve works on each channel independently if we iterate,
        # but it also supports multi-dimensional arrays. For a 2D kernel
        # applied to a 3-channel image, we need to pad the kernel dims.
        if image.ndim == 3 and kernel.ndim == 2:
            kernel_3d = kernel[:, :, _xp.newaxis]
            return convolve(image, kernel_3d)
        return convolve(image, kernel)
    else:
        return cv2.filter2D(image, -1, kernel)


# ---------------------------------------------------------------------------
# Hardware & video info display
# ---------------------------------------------------------------------------


def print_hardware_info():
    """Print hardware information based on current backend."""
    print("=" * 60)
    print("  Hardware Information")
    print("=" * 60)
    print(f"  OS           : {platform.system()} {platform.release()}")
    print(f"  Python       : {platform.python_version()}")
    print(f"  Accel Mode   : {_backend.upper()}")

    if _backend == "cuda":
        import cupy as cp
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        total_mem = props["totalGlobalMem"] / (1024 ** 3)
        print(f"  GPU          : {name}")
        print(f"  GPU Memory   : {total_mem:.1f} GB")
        print(f"  CUDA Version : {cp.cuda.runtime.runtimeGetVersion()}")
    else:
        try:
            import psutil
            cpu_name = platform.processor() or "Unknown"
            cores_phys = psutil.cpu_count(logical=False)
            cores_logic = psutil.cpu_count(logical=True)
            mem = psutil.virtual_memory()
            print(f"  CPU          : {cpu_name}")
            print(f"  CPU Cores    : {cores_phys} physical / {cores_logic} logical")
            print(f"  RAM          : {mem.total / (1024 ** 3):.1f} GB")
        except ImportError:
            print(f"  CPU          : {platform.processor() or 'Unknown'}")
            print(f"  CPU Cores    : {os.cpu_count()}")
        print(f"  Threads Used : {_threads}")

    print("=" * 60)


def print_video_info(video_path: str, images, fps: float):
    """Print input video parameters."""
    num_frames = images.shape[0]
    height, width = images.shape[1], images.shape[2]
    duration = num_frames / fps if fps > 0 else 0

    print()
    print("=" * 60)
    print("  Input Video Information")
    print("=" * 60)
    print(f"  File         : {os.path.basename(video_path)}")
    print(f"  Resolution   : {width} x {height}")
    print(f"  FPS          : {fps:.2f}")
    print(f"  Frames       : {num_frames}")
    print(f"  Duration     : {duration:.2f} s")
    print("=" * 60)
    print()
