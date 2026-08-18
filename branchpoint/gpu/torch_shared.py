"""
Torch backend: a wgpu buffer whose memory is shared with CUDA, written by
torch with a single device-to-device copy, plus a texture wrapper that blits
the buffer into a wgpu texture for pygfx to sample. Eliminates host round trip.

Flow: torch tensor --copy_--> SharedTensorBuffer --copy_buffer_to_texture--> wgpu texture --> pygfx
"""

import logging

import torch

from branchpoint import _native

logger = logging.getLogger(__name__)

try:
    from cuda.bindings import driver as cu
except ImportError:  # cuda-python < 12.6 layout
    from cuda import cuda as cu
import cupy as cp


def _ck(err, what):
    if isinstance(err, tuple):
        code, *rest = err
    else:
        code, rest = err, []
    if code != cu.CUresult.CUDA_SUCCESS:
        _, name = cu.cuGetErrorName(code)
        raise RuntimeError(f"CUDA error in {what}: {name}")
    return rest[0] if len(rest) == 1 else rest


def _ensure_cuda_ctx():
    _ck(cu.cuInit(0), "cuInit")
    torch.cuda.init()
    torch.zeros(1, device="cuda")  # force torch's primary context
    code, cur = cu.cuCtxGetCurrent()
    if code != cu.CUresult.CUDA_SUCCESS or int(cur) == 0:
        dev = _ck(cu.cuDeviceGet(0), "cuDeviceGet")
        ctx = _ck(cu.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
        _ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")


class SharedTensorBuffer:
    """A wgpu buffer aliased into CUDA, updatable from torch with one D2D copy.

    Parameters
    ----------
    device : wgpu-py GPUDevice (pygfx's shared device)
    shape : tuple[int, ...]  logical f32 shape of the shared region
    """

    def __init__(self, device, shape):
        _ensure_cuda_ctx()

        self.shape = tuple(shape)
        self.nbytes = 4 * int(torch.tensor(self.shape).prod())
        self._handle = _native.create_exportable_buffer(device, self.nbytes)
        self.gpu_buffer = _native.wrap_as_gpubuffer(self._handle, device)

        # CUDA import (consumes the fd)
        hdesc = cu.CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
        hdesc.type = (
            cu.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
        )
        hdesc.handle.fd = self._handle.fd
        hdesc.size = self._handle.alloc_size
        self._ext_mem = _ck(cu.cuImportExternalMemory(hdesc), "cuImportExternalMemory")

        bdesc = cu.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
        bdesc.offset, bdesc.size, bdesc.flags = 0, self.nbytes, 0
        dptr = _ck(
            cu.cuExternalMemoryGetMappedBuffer(ext_mem := self._ext_mem, bdesc),
            "cuExternalMemoryGetMappedBuffer",
        )

        self._umem = cp.cuda.UnownedMemory(int(dptr), self.nbytes, owner=None)
        carr = cp.ndarray(
            (self.nbytes // 4,),
            dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(self._umem, 0),
        )
        self.view = torch.from_dlpack(carr).view(
            *self.shape
        )  # torch view of shared mem
        self._event = torch.cuda.Event()
        self._closed = False

    def update(self, src: torch.Tensor) -> None:
        """Copy `src` into the shared region (fused cast/contiguity, one D2D copy)."""
        self.view.copy_(src)  # handles dtype cast + non-contiguous gather
        self._event.record()

    def sync(self) -> None:
        """Order CUDA writes before the next wgpu submit that reads the buffer."""
        self._event.synchronize()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # strict reverse ownership order (design doc §5)
        self.view = None
        self._umem = None
        _ck(cu.cuDestroyExternalMemory(self._ext_mem), "cuDestroyExternalMemory")
        self.gpu_buffer.destroy()
        dev = self.gpu_buffer._device
        self.gpu_buffer = None
        for _ in range(3):
            dev._poll()
        _native.free_exportable_memory(self._handle)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
