import os
from pathlib import Path

# until changes are integrated upstream, use vendored .so for wgpu-native
_SO = Path(__file__).parent / "_native_lib" / "libwgpu_native.so"
os.environ.setdefault("WGPU_LIB_PATH", str(_SO))

from .shared import TorchTensorTexture, TinygradTensorTexture
from .device import install

__version__ = "0.0.1"

__all__ = [
    "TinygradTensorTexture",
    "TorchTensorTexture",
    "__version__",
    "install",
]
