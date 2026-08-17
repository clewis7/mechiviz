import ctypes
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


#: wgpu-native version the vendored .so was built from.  Used both for the
#: vendored filename and for the runtime ABI check against wgpu-py.
WGPU_NATIVE_VERSION = (27, 0, 2, 0)

#: Vendored library location inside the package
_VENDORED_SO = (
        Path(__file__).parent
        / "_native_lib"
        / f"libwgpu_native-{'.'.join(map(str, WGPU_NATIVE_VERSION))}-exportable.so"
)

# Default usage for shared buffers: storage/vertex for consumption,
# copy-src/dst for the buffer->texture path and for debug readback.
# Values mirror webgpu.h WGPUBufferUsage flags
USAGE_COPY_SRC = 0x0004
USAGE_COPY_DST = 0x0008
USAGE_VERTEX = 0x0020
USAGE_STORAGE = 0x0080
DEFAULT_USAGE = USAGE_STORAGE | USAGE_VERTEX | USAGE_COPY_SRC | USAGE_COPY_DST




def ensure_lib_path() -> Optional[str]:
    """Point wgpu-py at the patched wgpu-native."""
    if "WGPU_LIB_PATH" in os.environ:
        path = os.environ["WGPU_LIB_PATH"]
        logger.debug("WGPU_LIB_PATH already set by user: %s", path)
        return path
    if _VENDORED_SO.exists():
        os.environ["WGPU_LIB_PATH"] = str(_VENDORED_SO)
        logger.debug("Using vendored patched wgpu-native: %s", _VENDORED_SO)
        return str(_VENDORED_SO)
    logger.debug("No patched wgpu-native found; stock library will be used")
    return None