import ctypes
import functools
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Vendored library location inside the package
VENDORED_SO = (
    Path(__file__).parent / "_native_lib" / "libwgpu_native-27.0.2.0-exportable.so"
)


# Default usage for shared buffers: storage/vertex for consumption,
# copy-src/dst for the buffer->texture path and for debug readback.
# Values mirror webgpu.h WGPUBufferUsage flags
USAGE_COPY_SRC = 0x0004
USAGE_COPY_DST = 0x0008
USAGE_VERTEX = 0x0020
USAGE_STORAGE = 0x0080
DEFAULT_USAGE = USAGE_STORAGE | USAGE_VERTEX | USAGE_COPY_SRC | USAGE_COPY_DST


if "wgpu" in sys.modules:
    logger.warning(
        "wgpu was imported before branchpoint — the patched wgpu-native "
        "cannot be loaded now. Import branchpoint before pygfx/fastplotlib/"
        "wgpu. Shared-memory backend will be unavailable this session."
    )
elif not VENDORED_SO.exists():
    logger.warning(
        "patched wgpu-native not found at %s — stock wgpu-native will be "
        "used and the shared-memory backend will be unavailable.",
        VENDORED_SO,
    )
else:
    os.environ["WGPU_LIB_PATH"] = str(VENDORED_SO)
    logger.debug("WGPU_LIB_PATH set to %s", VENDORED_SO)


@functools.cache
def _load() -> ctypes.CDLL:
    """dlopen the wgpu-native that wgpu-py is using; declare signatures."""
    import wgpu.backends.wgpu_native as wn  # deliberate late import

    logger.debug("loading wgpu-native from %s", wn.lib_path)
    cdll = ctypes.CDLL(wn.lib_path)

    try:
        fn = cdll.wgpuDeviceCreateExportableBuffer
    except AttributeError:
        logger.info(
            "wgpu-native at %s lacks the Branchpoint exportable-buffer "
            "symbols; shared-memory backend disabled (host-copy fallback)",
            wn.lib_path,
        )
        sys.exit()

    # WGPUBuffer wgpuDeviceCreateExportableBuffer(
    #     WGPUDevice, uint64_t size, uint64_t usage,
    #     int32_t* out_fd, uint64_t* out_alloc_size, uint64_t* out_memory)
    fn.restype = ctypes.c_void_p
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]

    free_fn = cdll.wgpuExportableBufferFreeMemory
    free_fn.restype = None
    free_fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

    return cdll


def _cdll() -> ctypes.CDLL:
    return _load()


@dataclass
class ExportableBufferHandle:
    """Raw result of ``wgpuDeviceCreateExportableBuffer``.

    Attributes
    ----------
    raw_buffer:
        Address of the ``WGPUBuffer`` (an ``Arc<WGPUBufferImpl>`` raw
        pointer).  Wrap it into a wgpu-py ``GPUBuffer`` via
        :func:`wrap_as_gpubuffer` before use; the wrapper then owns the
        release of this reference.
    fd:
        Exported opaque POSIX fd.  **Consumed** by
        ``cuImportExternalMemory`` — never close it manually after a
        successful import, never import it twice.
    alloc_size:
        The driver's ``VkMemoryRequirements.size``.  Pass THIS (not
        ``nbytes``) as ``CUDA_EXTERNAL_MEMORY_HANDLE_DESC.size``.
    vk_memory:
        Raw ``VkDeviceMemory`` handle.  Owned by the caller; release with
        :func:`free_exportable_memory` only after the wgpu buffer is fully
        released AND ``cuDestroyExternalMemory`` has run.
    nbytes / usage:
        The logical size and WGPUBufferUsage the buffer was created with
        (recorded for the wgpu-py wrapper).
    device_ptr:
        Address of the WGPUDevice, recorded for the paired free call.
    """

    raw_buffer: int
    fd: int
    alloc_size: int
    vk_memory: int
    nbytes: int
    usage: int
    device_ptr: int


def _device_address(device) -> int:
    """WGPUDevice address from a wgpu-py GPUDevice."""
    from wgpu.backends.wgpu_native._ffi import ffi

    return int(ffi.cast("uintptr_t", device._internal))


def create_exportable_buffer(
    device, nbytes: int, usage: int = DEFAULT_USAGE
) -> ExportableBufferHandle:
    """Create an exportable wgpu buffer on ``device`` (wgpu-py GPUDevice).

    Raises :class:`ExportableBufferError` if the patched symbol is missing
    or creation fails.  Check :func:`has_exportable` first to fall back
    cleanly.
    """
    if nbytes <= 0:
        raise ValueError("nbytes must be positive")

    device_ptr = _device_address(device)
    fd = ctypes.c_int32(-1)
    alloc_size = ctypes.c_uint64(0)
    vk_memory = ctypes.c_uint64(0)

    raw = _cdll().wgpuDeviceCreateExportableBuffer(
        device_ptr,
        nbytes,
        usage,
        ctypes.byref(fd),
        ctypes.byref(alloc_size),
        ctypes.byref(vk_memory),
    )

    return ExportableBufferHandle(
        raw_buffer=raw,
        fd=fd.value,
        alloc_size=alloc_size.value,
        vk_memory=vk_memory.value,
        nbytes=nbytes,
        usage=usage,
        device_ptr=device_ptr,
    )


def wrap_as_gpubuffer(
    handle: ExportableBufferHandle, device, label: str = "branchpoint-shared"
):
    """Wrap the raw WGPUBuffer as a first-class wgpu-py ``GPUBuffer``.

    The returned object works with ``queue.write_buffer``, bind groups,
    ``copy_buffer_to_texture``, etc.  Its garbage collection releases the
    underlying ``Arc`` reference (via ``wgpuBufferRelease``); the
    ``VkDeviceMemory`` still requires :func:`free_exportable_memory`
    afterwards.
    """
    from wgpu.backends.wgpu_native import _api as wnapi
    from wgpu.backends.wgpu_native._ffi import ffi

    internal = ffi.cast("WGPUBuffer", handle.raw_buffer)
    return wnapi.GPUBuffer(
        label, internal, device, handle.nbytes, handle.usage, "unmapped"
    )


def free_exportable_memory(handle: ExportableBufferHandle) -> None:
    """Free the VkDeviceMemory backing an exportable buffer.

    Preconditions (caller-enforced; violating them is a GPU use-after-free
    that no validation layer will catch):
      1. The GPUBuffer from :func:`wrap_as_gpubuffer` has been destroyed
         and released (call ``buf.destroy()``, drop references, then poll
         the device so the drop flushes).
      2. ``cuDestroyExternalMemory`` has run on the CUDA import.
    """
    if handle.vk_memory == 0:
        return
    _cdll().wgpuExportableBufferFreeMemory(handle.device_ptr, handle.vk_memory)
    handle.vk_memory = 0  # idempotence: double-free becomes a no-op
