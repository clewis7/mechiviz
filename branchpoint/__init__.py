from branchpoint import _native

from .gpu import install, TorchTensorTexture, TinygradTensorTexture

__version__ = "0.0.1"


def has_shared_memory_backend() -> bool:
    """True if the patched wgpu-native is active (torch zero host-copy path)."""
    return _native.has_exportable()


__all__ = [
    "__version__",
    "install",
    "TorchTensorTexture",
    "TinygradTensorTexture",
    "has_shared_memory_backend",
]
