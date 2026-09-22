import contextlib
import math
import os
import types

import torch
from cuda.bindings import driver as cu

from ..utils import _native


def _check(result):
    """Unwrap a cuda-python (err, *values) tuple, raising on error."""
    err, *values = result
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {cu.cuGetErrorName(err)[1].decode()}")
    return values[0] if len(values) == 1 else None


class SharedTensorBuffer:
    """A wgpu buffer aliased into CUDA, updatable from torch with one D2D copy.

    Parameters
    ----------
    device : wgpu-py GPUDevice (pygfx's shared device)
    shape : tuple[int, ...]  logical f32 shape of the shared region
    """

    def __init__(self, device, shape):
        # initializes CUDA and makes torch's primary context current
        torch.zeros(1, device="cuda")

        self.shape = tuple(shape)
        self.nbytes = 4 * math.prod(self.shape)

        eb = _native.create_exportable_buffer(device, self.nbytes)
        self.gpu_buffer = eb.buffer

        hdesc = cu.CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
        hdesc.type = (
            cu.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
        )
        hdesc.handle.fd = eb.fd
        hdesc.size = eb.alloc_size
        try:
            self._ext_mem = _check(cu.cuImportExternalMemory(hdesc))
        except RuntimeError:
            os.close(eb.fd)  # CUDA only takes ownership of the fd on success
            raise

        bdesc = cu.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
        bdesc.offset, bdesc.size = 0, self.nbytes
        self._dptr = _check(cu.cuExternalMemoryGetMappedBuffer(self._ext_mem, bdesc))

        # Zero-copy torch view of the shared memory via __cuda_array_interface__.
        cai = types.SimpleNamespace(
            __cuda_array_interface__={
                "data": (int(self._dptr), False),
                "shape": self.shape,
                "typestr": "<f4",
                "version": 3,
            }
        )
        self.view = torch.as_tensor(cai, device="cuda")
        self._event = torch.cuda.Event()

    def update(self, src: torch.Tensor) -> None:
        """Copy `src` into the shared region (fused cast/contiguity, one D2D copy)."""
        self.view.copy_(src)
        self._event.record()

    def sync(self) -> None:
        """Order CUDA writes before the next wgpu submit that reads the buffer."""
        self._event.synchronize()

    def close(self) -> None:
        if self.gpu_buffer is None:
            return
        self._event.synchronize()
        self.view = None
        _check(cu.cuMemFree(self._dptr))
        _check(cu.cuDestroyExternalMemory(self._ext_mem))
        self.gpu_buffer.destroy()  # wgpu frees the memory once released
        self.gpu_buffer = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
