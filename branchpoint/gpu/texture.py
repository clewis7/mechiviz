from typing import Any

import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object
from tinygrad import Tensor

from .transfer import padded_row_texels, copy_tensor_to_texture
from .device import DEFAULT_NAME, SharedWebGpuDevice, installed
from .torch_shared import SharedTensorBuffer


DEFAULT_USAGE = wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING


class _TensorTextureBase:
    """Fixed-size pygfx texture written by a compute framework on the GPU.

    Owns everything pygfx-facing: the gfx.Texture wrapper, lazy resolution of
    the raw wgpu texture, row-pitch padding math, prepare(), and as_image().
    Subclasses implement update() for specific framework.
    """

    def __init__(
        self, height: int, width: int, fmt: str = "r32float", usage: int = DEFAULT_USAGE
    ):
        self.height = int(height)
        self.width = int(width)
        self.fmt = fmt
        self.usage = usage

        self.row_texels = padded_row_texels(self.width, fmt)
        self.padded = self.row_texels != self.width

        self._gfx_texture: Any | None = None
        self._wgpu_texture: Any | None = None

    # TODO: make this a settable property such that an existing graphic with matching shape can be used to copy to
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


class TinygradTensorTexture(_TensorTextureBase):
    """tinygrad backend: true zero-copy via the shared wgpu device."""

    def __init__(
        self,
        height,
        width,
        fmt="r32float",
        dev: "SharedWebGpuDevice | None" = None,
        device_name: str = DEFAULT_NAME,
        usage: int = DEFAULT_USAGE,
    ):
        super().__init__(height, width, fmt, usage)
        if dev is None:
            dev = installed(device_name)
            if dev is None:
                raise RuntimeError(
                    "no shared device installed. Call branchpoint.gpu.install() "
                    "(after creating your renderer, before creating tensors) first."
                )
        self.dev = dev
        self._staging = None

    def update(self, t, synchronize: bool = True) -> None:
        self._check_shape(t.shape)
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

    def __init__(
        self, height, width, fmt="r32float", device=None, usage: int = DEFAULT_USAGE
    ):
        if fmt not in self.BYTES_PER_TEXEL:
            raise ValueError(
                f"torch backend supports {list(self.BYTES_PER_TEXEL)} for now"
            )
        super().__init__(height, width, fmt, usage)
        if device is None:
            device = gfx.renderers.wgpu.get_shared().device  # pygfx's wgpu device
        self.device = device
        self.buf = SharedTensorBuffer(device, (self.height, self.row_texels))

    def update(self, src, synchronize: bool = True) -> None:
        self._check_shape(src.shape[-2:])
        self.buf.view[:, : self.width].copy_(src)  # cast+contiguity+D2D in one
        self.buf._event.record()
        if synchronize:
            self.buf.sync()  # v1 coarse sync: CUDA done before the blit reads

        bpt = self.BYTES_PER_TEXEL[self.fmt]
        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_texture(
            {
                "buffer": self.buf.gpu_buffer,
                "offset": 0,
                "bytes_per_row": bpt * self.row_texels,
                "rows_per_image": self.height,
            },
            {"texture": self._resolve(), "mip_level": 0, "origin": (0, 0, 0)},
            (self.width, self.height, 1),
        )
        self.device.queue.submit([enc.finish()])

    def close(self):
        self.buf.close()
