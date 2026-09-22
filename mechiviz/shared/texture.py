from typing import Any, Sequence

import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object
from tinygrad import Tensor

from utils.transfer import padded_row_texels, copy_tensor_to_texture, bytes_per_texel
from device.tinygrad_shared import DEFAULT_NAME, SharedWebGpuDevice, installed
from device.torch_shared import SharedTensorBuffer


DEFAULT_USAGE = wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING

_CHANNELS = {
    "r32float": 1,
    "r8unorm": 1,
    "rg32float": 2,
    "rgba32float": 4,
    "rgba8unorm": 4,
    "rgba8uint": 4,
}

_FORMAT_FOR_CHANNELS = {
    1: "r32float",
    2: "rg32float",
    3: "rgba32float",  # no 3-channel format exists; pad to 4
    4: "rgba32float",
}


class _TensorTextureBase:
    """Fixed-size pygfx texture written by a compute framework on the GPU.

    Owns everything pygfx-facing: the gfx.Texture wrapper, lazy resolution of
    the raw wgpu texture, row-pitch padding math, prepare(), and as_image().
    Subclasses implement update() for specific framework.
    """

    def __init__(self, shape: Sequence[int], usage: int = DEFAULT_USAGE):
        h, w, c = self._parse_shape(shape)
        self.height, self.width = int(h), int(w)
        self.usage = usage

        self._gfx_texture: Any | None = None
        self._wgpu_texture: Any | None = None

        # sets fmt/bytes_per_texel/n_channels/row_texels, then calls _alloc()
        self._apply_format(_FORMAT_FOR_CHANNELS[c])

    @staticmethod
    def _parse_shape(shape: Sequence[int]):
        if len(shape) == 2:
            (h, w), c = shape, 1
        elif len(shape) == 3:
            h, w, c = shape
        else:
            raise ValueError(f"shape must be (H, W) or (H, W, C), got {shape}")

        if c not in _FORMAT_FOR_CHANNELS.keys():
            raise ValueError(f"channels must be 1-4, got {c}")
        return h, w, c

    def _apply_format(self, fmt: str) -> None:
        """Set the format and everything derived from it, then (re)allocate."""
        if fmt not in _CHANNELS:
            raise ValueError(
                f"unsupported texture format {fmt!r}; "
                f"this class handles {sorted(_CHANNELS)}"
            )
        self.fmt = fmt
        self.bytes_per_texel = bytes_per_texel(fmt)
        self.n_channels = _CHANNELS[fmt]
        self.row_texels = padded_row_texels(self.width, self.fmt)
        self.padded = self.row_texels != self.width
        self._alloc()  # backend-specific

    def _alloc(self) -> None:
        raise NotImplementedError

    @property
    def format(self) -> str:
        return self.fmt

    @property
    def texture(self):
        """The `gfx.Texture` to hand to a pygfx Geometry."""
        if self._gfx_texture is None:
            self._gfx_texture = gfx.Texture(
                size=(self.width, self.height, 1),
                dim=2,
                format=self.fmt,
                usage=self.usage,
            )
        return self._gfx_texture

    @texture.setter
    def texture(self, tex):
        """Adopt an existing gfx.Texture (e.g. from a fastplotlib graphic)."""
        if tex is None:
            self._gfx_texture = None
            self._wgpu_texture = None
            return

        if not isinstance(tex, gfx.Texture):
            raise TypeError(f"expected gfx.Texture, got {type(tex).__name__}")

        self._gfx_texture = tex
        self._wgpu_texture = None

        raw = self._resolve()
        w, h = raw.size[0], raw.size[1]
        if (h, w) != (self.height, self.width):
            raise ValueError(
                f"texture is {h}x{w} (h x w), this wrapper is "
                f"{self.height}x{self.width}"
            )

        self._apply_format(str(raw.format))

    def _resolve(self):
        """Get the underlying wgpu.GPUTexture, materializing it if needed."""
        if self._wgpu_texture is None:
            raw = ensure_wgpu_object(self.texture)
            if raw is None:
                raise RuntimeError(
                    "pygfx has not created the GPU texture yet. Add this "
                    "texture to a scene and render one frame before calling "
                    "update() - or call prepare() from inside your draw "
                    "callback rather than during setup."
                )
            self._wgpu_texture = raw
        return self._wgpu_texture

    def prepare(self) -> None:
        """Force wgpu texture resolution."""
        self._resolve()

    def as_image(
        self,
        clim=(0.0, 1.0),
        cmap=None,
        position=(0.0, 0.0, 0.0),
        scale=1.0,
        flip_y: bool = True,
        interpolation: str = "nearest",
    ):
        """Get the underlying texture as an image that can be rendered."""
        kwargs: dict[str, Any] = {"clim": clim, "interpolation": interpolation}
        if cmap is not None:
            kwargs["map"] = cmap
        img = gfx.Image(
            gfx.Geometry(grid=self.texture), gfx.ImageBasicMaterial(**kwargs)
        )
        sx, sy = (scale, scale) if isinstance(scale, (int, float)) else scale
        x, y, z = position
        if flip_y:
            img.local.position = (x, y + self.height * sy, z)
            img.local.scale = (sx, -sy, 1)
        else:
            img.local.position = (x, y, z)
            img.local.scale = (sx, sy, 1)
        return img

    def _check_shape(self, shape) -> None:
        """Ensure shape of tensor being updated matches expected shape."""
        if tuple(shape) != (self.height, self.width):
            raise ValueError(
                f"this texture is {self.height}x{self.width}, got a tensor of "
                f"shape {tuple(shape)}."
            )

    def _src_channels(self, src) -> int:
        """Validate an update source and return how many channels it carries."""
        shape = tuple(src.shape)
        if len(shape) == 2:
            c = 1
        elif len(shape) == 3:
            c = shape[2]
        else:
            raise ValueError(f"expected (H, W) or (H, W, C), got shape {shape}")

        self._check_shape(shape[:2])
        if c > self.n_channels:
            raise ValueError(
                f"source has {c} channels but texture format {self.fmt} "
                f"stores {self.n_channels}."
            )
        return c


class TinygradTensorTexture(_TensorTextureBase):
    """tinygrad backend: true zero-copy via the shared wgpu device."""

    def __init__(
        self,
        shape,
        dev: "SharedWebGpuDevice | None" = None,
        device_name: str = DEFAULT_NAME,
        usage: int = DEFAULT_USAGE,
    ):
        if dev is None:
            dev = installed(device_name)
            if dev is None:
                raise RuntimeError(
                    "no shared device installed. Call mechiviz.gpu.install() "
                    "(after creating your renderer, before creating tensors) first."
                )
        self.dev = dev
        self._staging = None
        super().__init__(shape, usage)

    def _alloc(self) -> None:
        # Staging is allocated lazily on first padded update; a format change
        # just invalidates it so the next update rebuilds at the right width.
        self._staging = None

    def update(self, t, synchronize: bool = True) -> None:
        c = self._src_channels(t)
        if c != self.n_channels:
            raise NotImplementedError(
                f"tinygrad backend currently handles {self.n_channels}-channel "
                f"data for format {self.fmt}; got {c} channels."
            )
        if self.fmt.endswith("float") and t.dtype.name != "float":
            t = t.float()
        src = self._write_staging(t) if self.padded else t.contiguous().realize()
        copy_tensor_to_texture(
            self.dev,
            src,
            self._resolve(),
            width=self.width,
            height=self.height,
            fmt=self.fmt,
            synchronize=synchronize,
        )

    def _write_staging(self, t):
        if self._staging is None:
            self._staging = (
                Tensor.zeros(self.height, self.row_texels, device=self.dev.device)
                .contiguous()
                .realize()
            )
        self._staging[:, : self.width] = t
        return self._staging.contiguous().realize()


class TorchTensorTexture(_TensorTextureBase):
    """torch backend: CUDA/Vulkan shared buffer + GPU-side blit into the texture."""

    BYTES_PER_TEXEL = {"r32float": 4}

    def __init__(self, shape, device=None, usage: int = DEFAULT_USAGE):
        if device is None:
            device = gfx.renderers.wgpu.get_shared().device  # pygfx's wgpu device
        self.device = device
        self.buf = None
        self.view = None
        super().__init__(shape, usage)

    def _alloc(self) -> None:
        """(Re)allocate the shared buffer for the current format."""
        if self.buf is not None:
            self.buf.close()

        # flat scalars: one row is row_texels * n_channels float32 values
        self.buf = SharedTensorBuffer(
            self.device, (self.height, self.row_texels * self.n_channels)
        )
        # channel-shaped view over the same memory
        self.view = self.buf.view.view(self.height, self.row_texels, self.n_channels)
        # Fill once, not per frame: any channel the source doesn't carry (alpha,
        # typically) keeps this value for the buffer's lifetime.
        self.view.fill_(1.0)

    def update(self, src, synchronize: bool = True) -> None:
        c = self._src_channels(src)
        if src.ndim == 2:
            src = src[..., None]

        self.view[:, : self.width, :c].copy_(src)  # cast+contiguity+D2D in one
        self.buf._event.record()
        if synchronize:
            self.buf.sync()  # v1 coarse sync: CUDA done before the blit reads

        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_texture(
            {
                "buffer": self.buf.gpu_buffer,
                "offset": 0,
                "bytes_per_row": self.bytes_per_texel * self.row_texels,
                "rows_per_image": self.height,
            },
            {"texture": self._resolve(), "mip_level": 0, "origin": (0, 0, 0)},
            (self.width, self.height, 1),
        )
        self.device.queue.submit([enc.finish()])

    def close(self):
        if self.buf is not None:
            self.buf.close()
            self.buf = None
            self.view = None
