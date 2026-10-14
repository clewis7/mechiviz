from typing import Any, Sequence

import numpy as np
import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object

from ..device import SharedTensorBuffer


DEFAULT_USAGE = (
    wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE
)

_FORMAT_FOR_CHANNELS = {1: "f4", 2: "2xf4", 3: "3xf4", 4: "4xf4"}


class _TensorBufferBase:
    """Fixed-size pygfx vertex buffer written by a compute framework on the GPU.

    The buffer analog of `_TensorTextureBase`: owns everything pygfx-facing,
    the gfx.Buffer wrapper, lazy resolution of the raw wgpu buffer, and
    prepare(). Subclasses implement update() for a specific framework.
    """

    def __init__(self, shape: Sequence[int], usage: int = DEFAULT_USAGE):
        n, c = self._parse_shape(shape)
        self.nitems, self.n_channels = int(n), int(c)
        self.usage = usage

        self._gfx_buffer: Any | None = None
        self._wgpu_buffer: Any | None = None

        self._apply_channels(self.n_channels)

    @staticmethod
    def _parse_shape(shape: Sequence[int]):
        if len(shape) == 1:
            (n,), c = shape, 1
        elif len(shape) == 2:
            n, c = shape
        else:
            raise ValueError(f"shape must be (N,) or (N, C), got {shape}")

        if c not in _FORMAT_FOR_CHANNELS:
            raise ValueError(f"channels must be 1-4, got {c}")
        return n, c

    def _apply_channels(self, c: int) -> None:
        """Set the channel count and everything derived from it, then (re)allocate."""
        self.n_channels = int(c)
        self.fmt = _FORMAT_FOR_CHANNELS[self.n_channels]
        self.itemsize = 4 * self.n_channels
        self.nbytes = self.nitems * self.itemsize
        self._alloc()  # backend-specific

    def _alloc(self) -> None:
        raise NotImplementedError

    @property
    def format(self) -> str:
        return self.fmt

    @property
    def buffer(self):
        """The `gfx.Buffer` to hand to a pygfx Geometry."""
        if self._gfx_buffer is None:
            data = np.zeros((self.nitems, self.n_channels), dtype=np.float32)
            self._gfx_buffer = gfx.Buffer(data, usage=self.usage, force_contiguous=True)
        return self._gfx_buffer

    @buffer.setter
    def buffer(self, buf):
        """Adopt an existing gfx.Buffer (e.g. from a fastplotlib graphic)."""
        if buf is None:
            self._gfx_buffer = None
            self._wgpu_buffer = None
            return

        if not isinstance(buf, gfx.Buffer):
            raise TypeError(f"expected gfx.Buffer, got {type(buf).__name__}")

        n, c = self._buffer_shape(buf)
        if n != self.nitems:
            raise ValueError(
                f"buffer has {n} items, this wrapper is sized for {self.nitems}"
            )

        self._gfx_buffer = buf
        self._wgpu_buffer = None

        for attr in ("_wgpu_usage", "usage"):
            if hasattr(buf, attr):
                try:
                    setattr(buf, attr, getattr(buf, attr) | self.usage)
                except AttributeError:
                    pass
                break

        if c != self.n_channels:
            self._apply_channels(c)

    @staticmethod
    def _buffer_shape(buf) -> tuple[int, int]:
        data = getattr(buf, "data", None)
        if data is not None and getattr(data, "ndim", 0) == 2:
            return int(data.shape[0]), int(data.shape[1])
        fmt = str(buf.format)
        c = int(fmt.split("x")[0]) if "x" in fmt else 1
        return int(buf.nitems), c

    def _resolve(self):
        """Get the underlying wgpu.GPUBuffer, materializing it if needed."""
        if self._wgpu_buffer is None:
            raw = ensure_wgpu_object(self.buffer)
            if raw is None:
                raise RuntimeError(
                    "pygfx has not created the GPU buffer yet. Add this buffer "
                    "to a scene and render one frame before calling update() - "
                    "or call prepare() from inside your draw callback rather "
                    "than during setup."
                )
            self._wgpu_buffer = raw
        return self._wgpu_buffer

    def prepare(self) -> None:
        """Force wgpu buffer resolution."""
        self._resolve()

    def as_line(self, colors=(1.0, 1.0, 1.0, 1.0), thickness: float = 1.0, **material):
        """Get the underlying buffer as a line that can be rendered."""
        if self.n_channels != 3:
            raise ValueError(
                f"as_line() needs 3 channels (x, y, z), this buffer has "
                f"{self.n_channels}"
            )
        mat = gfx.LineMaterial(thickness=thickness, color=colors, **material)
        return gfx.Line(gfx.Geometry(positions=self.buffer), mat)

    def _check_shape(self, shape) -> None:
        """Ensure shape of tensor being updated matches expected shape."""
        shape = tuple(shape)
        if len(shape) == 1:
            shape = (shape[0], 1)
        if shape != (self.nitems, self.n_channels):
            raise ValueError(
                f"this buffer is {self.nitems}x{self.n_channels}, got a tensor "
                f"of shape {shape}."
            )


class TorchTensorBuffer(_TensorBufferBase):
    """torch backend: CUDA/Vulkan shared buffer + GPU-side blit into the vertex buffer."""

    def __init__(self, shape, device=None, usage: int = DEFAULT_USAGE):
        if device is None:
            device = gfx.renderers.wgpu.get_shared().device  # pygfx's wgpu device
        self.device = device
        self.buf = None
        self.view = None
        super().__init__(shape, usage)

    def _alloc(self) -> None:
        """(Re)allocate the shared buffer for the current channel count."""
        if self.buf is not None:
            self.buf.close()

        self.buf = SharedTensorBuffer(self.device, (self.nitems, self.n_channels))
        self.view = self.buf.view
        # Fill once, not per frame: any channel the source doesn't carry (alpha,
        # typically) keeps this value for the buffer's lifetime.
        self.view.fill_(1.0)

    def update(self, src, synchronize: bool = True) -> None:
        """Copy a torch tensor into the vertex buffer without touching the host."""
        if src.ndim == 1:
            src = src[:, None]
        self._check_shape(src.shape[:1] + (src.shape[1],))
        c = src.shape[1]

        self.view[:, :c].copy_(src)  # cast + contiguity + D2D in one
        self.buf._event.record()
        if synchronize:
            self.buf.sync()  # v1 coarse sync: CUDA done before the blit reads

        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_buffer(
            self.buf.gpu_buffer, 0, self._resolve(), 0, self.nbytes
        )
        self.device.queue.submit([enc.finish()])

    def close(self):
        if self.buf is not None:
            self.buf.close()
            self.buf = None
            self.view = None
