from typing import Any

import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object
from tinygrad import Tensor
import numpy as np

from .transfer import padded_row_texels, copy_tensor_to_texture, copy_tensor_to_buffer
from .device import DEFAULT_NAME, SharedWebGpuDevice, installed


DEFAULT_USAGE = wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING


class TensorTexture:
    """A fixed-size pygfx texture fed from tinygrad Tensors on the same device.

    Args:
        height, width: extent in texels, in TENSOR order (rows first).
        fmt: wgpu texture format. Defaults to single-channel float32
        dev: the SharedWebGpuDevice
    """

    def __init__(
        self,
        height: int,
        width: int,
        fmt: str = "r32float",
        dev: SharedWebGpuDevice | None = None,
        device_name: str = DEFAULT_NAME,
        usage: int = DEFAULT_USAGE,
    ):
        if dev is None:
            dev = installed(device_name)
            if dev is None:
                raise RuntimeError(
                    "no shared device installed. Call "
                    "branchpoint.gpu.install() (after creating your "
                    "renderer, before creating tensors) first."
                )
        self.dev = dev
        self.height = int(height)
        self.width = int(width)
        self.fmt = fmt
        self.usage = usage

        # pad if needed
        self.row_texels = padded_row_texels(self.width, fmt)
        self.padded = self.row_texels != self.width

        self._gfx_texture: Any | None = None  # the pygfx wrapper
        self._wgpu_texture: Any | None = None  # the real GPU object, resolved lazily
        self._staging = None  # allocated on first padded update

    @property
    def texture(self):
        """The `gfx.Texture` to hand to a pygfx Geometry."""
        if self._gfx_texture is None:
            self._gfx_texture = self._make_texture()
        return self._gfx_texture

    def _make_texture(self):
        """Create the pygfx texture"""
        # pygfx/wgpu size is (width, height, depth)
        return gfx.Texture(
            size=(self.width, self.height, 1),
            dim=2,
            format=self.fmt,
            usage=self.usage,
        )

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
        """Build a `gfx.Image` displaying this texture.

        clim fixes the displayed value range. PREFER A FIXED RANGE: normalizing
        per frame from the data's own min/max makes brightness incomparable
        between frames, which defeats the purpose when the thing you are
        watching for is a pattern getting stronger. Attention weights are
        already in [0, 1], so the default is usually right.

        scale may be a number or an (sx, sy) pair - the pair is what you want
        for wide, short strips such as a per-position loss bar.

        flip_y puts row 0 at the top, which is how attention matrices are read.
        gfx.Image anchors at a corner and spans +y, so a negative y-scale
        mirrors about position.y; the offset below compensates.
        """
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

    def update(self, t, synchronize: bool = True) -> None:
        """Copy tensor `t` into the texture, entirely on the GPU.

        `t` must have shape (height, width). A mismatch raises.
        """
        shape = tuple(t.shape)
        if shape != (self.height, self.width):
            raise ValueError(
                f"this texture is {self.height}x{self.width}, got a tensor of "
                f"shape {shape}. Reshape it, or make a TensorTexture of that "
                f"size instead."
            )

        if self.fmt.endswith("float") and t.dtype.name != "float":
            t = t.float()

        if self.padded:
            # Write the real data into the left part of a row-aligned staging
            # tensor. The pad columns keep whatever they had; the copy extent
            # below never reads them.
            src = self._write_staging(t)
        else:
            # Fast path: the tensor's own rows are already legally aligned, so
            # copy directly out of its buffer. No staging, no allocation.
            src = t.contiguous().realize()

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
        """Assign `t` into the persistent row-aligned staging tensor.

        Allocated once and assigned into thereafter, so the underlying wgpu
        buffer handle stays stable and the frame loop does not churn the
        allocator.
        """
        if self._staging is None:
            # Compiled stores the device name as `.device` (not `.dname`).
            self._staging = (
                Tensor.zeros(self.height, self.row_texels, device=self.dev.device)
                .contiguous()
                .realize()
            )
        self._staging[:, : self.width] = t
        return self._staging.contiguous().realize()


class TensorBuffer:
    """A pygfx vertex buffer fed from tinygrad Tensors on the same device.

    The sibling of TensorTexture, for line data rather than image data."""

    def __init__(
        self,
        n_items: int,
        components: int = 3,
        dev: SharedWebGpuDevice | None = None,
        device_name: str = DEFAULT_NAME,
    ):
        if dev is None:
            dev = installed(device_name)
            if dev is None:
                raise RuntimeError(
                    "no shared device installed. Call "
                    "branchpoint.gpu.install_from_pygfx() first."
                )
        self.dev = dev
        self.n_items = int(n_items)
        self.components = int(components)
        self._gfx_buffer: Any | None = None
        self._wgpu_buffer: Any | None = None

    @property
    def buffer(self):
        """The `gfx.Buffer` to hand to a pygfx Geometry."""
        if self._gfx_buffer is None:
            self._gfx_buffer = self._make_buffer()
        return self._gfx_buffer

    def _make_buffer(self):
        # NaN positions are skipped by pygfx's line renderer, so an unfilled
        # buffer draws nothing rather than a spray of points at the origin.
        data = np.full((self.n_items, self.components), np.nan, np.float32)
        if self.components >= 3:
            data[:, 2] = 0.0
        # COPY_DST is what makes this buffer a legal copy destination. pygfx
        # does not set it on its own and it cannot be added later.
        return gfx.Buffer(data, usage=wgpu.BufferUsage.COPY_DST)

    def _resolve(self):
        if self._wgpu_buffer is None:
            raw = ensure_wgpu_object(self.buffer)
            if raw is None:
                raise RuntimeError(
                    "pygfx has not created the GPU buffer yet. Render one frame "
                    "before calling update()."
                )
            self._wgpu_buffer = raw
        return self._wgpu_buffer

    def prepare(self) -> None:
        self._resolve()

    def as_line(self, color="#ffffff", thickness: float = 2.0, **material_kwargs):
        """Build a `gfx.Line` drawing this buffer's positions."""
        return gfx.Line(
            gfx.Geometry(positions=self.buffer),
            gfx.LineMaterial(color=color, thickness=thickness, **material_kwargs),
        )

    def as_points(self, color="#ffffff", size: float = 4.0, **material_kwargs):
        """Build a `gfx.Points` drawing this buffer's positions."""
        import pygfx as gfx

        return gfx.Points(
            gfx.Geometry(positions=self.buffer),
            gfx.PointsMaterial(color=color, size=size, **material_kwargs),
        )

    def update(self, t, synchronize: bool = True) -> None:
        """Copy tensor `t`, shape (n_items, components) float32, into the buffer."""
        shape = tuple(t.shape)
        want = (self.n_items, self.components)
        if shape != want:
            raise ValueError(f"this buffer holds {want}, got a tensor of shape {shape}")
        if t.dtype.name != "float":
            t = t.float()

        copy_tensor_to_buffer(
            self.dev, t.contiguous().realize(), self._resolve(), synchronize=synchronize
        )


def pack_rgba8(t, lo=None, hi=None):
    """Normalize a 2D tensor to [0,1] and expand to rgba8 greyscale.

    Only needed for a renderer that cannot sample float textures. Prefer an
    r32float TensorTexture with a fixed clim: this function bakes display
    decisions into the data and costs extra kernels every frame.

    lo/hi fix the normalization range. Leaving them None uses per-frame
    min/max, which makes faint structure visible early but means brightness is
    not comparable across frames.
    """
    mn = t.min() if lo is None else lo
    mx = t.max() if hi is None else hi
    n = ((t - mn) / (mx - mn + 1e-8)).clip(0, 1)
    g = (n * 255).cast("uint8")
    a = Tensor.full(tuple(t.shape), 255, dtype="uint8", device=t.device)
    return Tensor.stack(g, g, g, a, dim=-1).contiguous()
