from typing import NamedTuple

import cffi
import wgpu
import wgpu.backends.wgpu_native as wn
from wgpu.backends.wgpu_native import _api
from wgpu.backends.wgpu_native._ffi import ffi as wgpu_ffi

DEFAULT_USAGE = (
    wgpu.BufferUsage.STORAGE
    | wgpu.BufferUsage.VERTEX
    | wgpu.BufferUsage.COPY_SRC
    | wgpu.BufferUsage.COPY_DST
)

# Reuse wgpu-py's type definitions so its cdata (devices, buffers) passes straight through.
_ffi = cffi.FFI()
_ffi.include(wgpu_ffi)
_ffi.cdef(
    "WGPUBuffer wgpuDeviceCreateExportableBuffer(WGPUDevice device, "
    "WGPUBufferDescriptor const * descriptor, int * fd, uint64_t * allocationSize);"
)
_lib = _ffi.dlopen(wn.lib_path)


class ExportableBuffer(NamedTuple):
    buffer: _api.GPUBuffer  # releasing it frees the memory
    fd: int  # owned by cuImportExternalMemory after a successful import
    alloc_size: int  # pass as CUDA_EXTERNAL_MEMORY_HANDLE_DESC.size


def create_exportable_buffer(
    device, nbytes: int, usage=DEFAULT_USAGE, label: str = "mechiviz-shared"
) -> ExportableBuffer:
    if getattr(_lib, "wgpuDeviceCreateExportableBuffer", None) is None:
        raise RuntimeError(
            f"wgpu-native at {wn.lib_path} lacks wgpuDeviceCreateExportableBuffer; "
            "import mechiviz before wgpu/pygfx/fastplotlib"
        )

    label_bytes = label.encode()
    c_label = _ffi.new("char[]", label_bytes)
    desc = _ffi.new(
        "WGPUBufferDescriptor *",
        {
            "label": {"data": c_label, "length": len(label_bytes)},
            "usage": int(usage),
            "size": nbytes,
        },
    )
    fd = _ffi.new("int *", -1)
    alloc_size = _ffi.new("uint64_t *")

    raw = getattr(_lib, "wgpuDeviceCreateExportableBuffer", None)(
        device._internal, desc, fd, alloc_size
    )
    if raw == _ffi.NULL:
        raise RuntimeError("exportable buffer creation failed; see wgpu-native log")

    buffer = _api.GPUBuffer(label, raw, device, nbytes, int(usage), "unmapped")
    if fd[0] < 0:
        raise RuntimeError("buffer created but memory export failed")

    return ExportableBuffer(buffer, fd[0], alloc_size[0])
