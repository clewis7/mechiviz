"""
Torch backend: a wgpu buffer whose memory is shared with CUDA, written by
torch with a single device-to-device copy, plus a texture wrapper that blits
the buffer into a wgpu texture for pygfx to sample, eliminating host round trip.

Flow: torch tensor --copy_--> SharedTensorBuffer --copy_buffer_to_texture--> wgpu texture --> pygfx
"""

import torch
import cuda.bindings.driver as cu
import cupy as cp

from branchpoint import _native


class SharedTensorBuffer:
    """A wgpu buffer aliased into CUDA, updatable from torch with one D2D copy.

    Parameters
    ----------
    device : wgpu-py GPUDevice (pygfx's shared device)
    shape : tuple[int, ...]  logical f32 shape of the shared region
    """

    def __init__(self, device, shape):
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

        _, self._ext_mem = cu.cuImportExternalMemory(hdesc)

        bdesc = cu.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
        bdesc.offset, bdesc.size, bdesc.flags = 0, self.nbytes, 0
        _, dptr = cu.cuExternalMemoryGetMappedBuffer(self._ext_mem, bdesc)

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
        # strict reverse ownership order
        self.view = None
        self._umem = None
        cu.cuDestroyExternalMemory(self._ext_mem)
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
