"""Make tinygrad and renderer use the same shared device."""

import functools
import struct
from typing import Any

from tinygrad.device import Allocator, BufferSpec, Compiled
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.helpers import DEV
from tinygrad.device import Device

from pygfx.renderers.wgpu import get_shared

# try importing needed wgpu enums
try:
    from wgpu import BufferUsage as _BU
    from wgpu import MapMode as _MM
    from wgpu import ShaderStage as _SS

    BUF_STORAGE = int(_BU.STORAGE)
    BUF_UNIFORM = int(_BU.UNIFORM)
    BUF_COPY_DST = int(_BU.COPY_DST)
    BUF_COPY_SRC = int(_BU.COPY_SRC)
    BUF_MAP_READ = int(_BU.MAP_READ)
    MAP_READ = int(_MM.READ)
    STAGE_COMPUTE = int(_SS.COMPUTE)
except Exception:
    BUF_STORAGE, BUF_UNIFORM = 0x080, 0x040
    BUF_COPY_DST, BUF_COPY_SRC = 0x008, 0x004
    BUF_MAP_READ = 0x001
    MAP_READ = 0x0001
    STAGE_COMPUTE = 0x4

BIND_STORAGE = "storage"
BIND_UNIFORM = "uniform"

DEFAULT_NAME = "WEBGPU"


def round_up(x: int, a: int) -> int:
    """Smallest multiple of `a` that is >= x. Used for buffer size alignment."""
    return (x + a - 1) // a * a


class SharedWebGPUProgram:
    """Compiles one WGSL kernel, dispatched against the shared device.

    The binding layout mirrors tinygrad's own ops_webgpu convention:
        binding 0             -> uniform f32 holding +inf
        bindings 1..nbufs     -> storage buffers, i.e. the kernel arguments
        bindings nbufs+1..    -> uniform i32/f32 scalars (tinygrad's `vals`)
    """

    def __init__(
        self, dev: "SharedWebGpuDevice", name: str, lib: bytes, *args, **kwargs
    ):
        # Newer tinygrad versions pass extra metadata (runtimevars=, prg=, ...)
        # to the program constructor. The WGSL dispatch path needs none of it,
        # accept and ignore anything past dev/name/lib
        self.dev = dev
        self.name = name
        self.src = lib.decode()
        self.module = dev.wdev.create_shader_module(code=self.src)

    def _uniform(self, val: int | float):
        """Allocate a 4-byte uniform buffer holding a single scalar."""
        d = self.dev.wdev
        b = d.create_buffer(size=4, usage=BUF_UNIFORM | BUF_COPY_DST)
        data = (
            val.to_bytes(4, "little")
            if isinstance(val, int)
            else struct.pack("<f", val)
        )
        d.queue.write_buffer(b, 0, data)
        return b

    def __call__(
        self,
        *bufs,
        global_size=(1, 1, 1),
        local_size=(1, 1, 1),
        vals=(),
        wait=False,
        **kwargs,
    ):
        d = self.dev.wdev
        nb = len(bufs)

        # --- bind group layout: describes the *shape* of the bindings ---
        entries_l: list[dict[str, Any]] = [
            {
                "binding": 0,
                "visibility": STAGE_COMPUTE,
                "buffer": {"type": BIND_UNIFORM},
            }
        ]
        for i in range(nb + len(vals)):
            entries_l.append(
                {
                    "binding": i + 1,
                    "visibility": STAGE_COMPUTE,
                    "buffer": {"type": BIND_STORAGE if i < nb else BIND_UNIFORM},
                }
            )
        bgl = d.create_bind_group_layout(entries=entries_l)
        pl = d.create_pipeline_layout(bind_group_layouts=[bgl])
        pipe = d.create_compute_pipeline(
            layout=pl, compute={"module": self.module, "entry_point": self.name}
        )

        # --- bind group: the actual buffers for this dispatch ---
        entries_b: list[dict[str, Any]] = [
            {
                "binding": 0,
                "resource": {
                    "buffer": self._uniform(float("inf")),
                    "offset": 0,
                    "size": 4,
                },
            }
        ]
        for i, b in enumerate(bufs):
            entries_b.append(
                {
                    "binding": i + 1,
                    "resource": {"buffer": b, "offset": 0, "size": b.size},
                }
            )
        for j, v in enumerate(vals):
            entries_b.append(
                {
                    "binding": nb + 1 + j,
                    "resource": {"buffer": self._uniform(v), "offset": 0, "size": 4},
                }
            )
        bg = d.create_bind_group(layout=bgl, entries=entries_b)

        enc = d.create_command_encoder()
        cpass = enc.begin_compute_pass()
        cpass.set_pipeline(pipe)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups(*global_size)
        cpass.end()
        d.queue.submit([enc.finish()])
        return None


class SharedWebGpuAllocator(Allocator):
    """Backs every tinygrad Buffer with a wgpu-py GPUBuffer on the shared device."""

    def _alloc(self, size: int, options: BufferSpec):
        return self.dev.wdev.create_buffer(
            # WebGPU requires 4-byte-aligned buffer sizes.
            size=round_up(size, 4),
            usage=BUF_STORAGE | BUF_COPY_DST | BUF_COPY_SRC,
        )

    def _copyin(self, dest, src: memoryview):
        mv = src
        if src.nbytes % 4:
            # write_buffer also wants a 4-byte-aligned length; pad the tail.
            pad = bytearray(round_up(src.nbytes, 4))
            pad[: src.nbytes] = src
            mv = memoryview(pad)
        self.dev.wdev.queue.write_buffer(dest, 0, mv)

    def _copyout(self, dest: memoryview, src):
        data = self.dev.read_buffer(src)
        dest[:] = data[: dest.nbytes]  # trim the alignment padding

    def _free(self, opaque, options: BufferSpec):
        # The renderer may have already torn the device down at interpreter exit.
        # Destroying a buffer whose device is gone corrupts memory in wgpu-native
        if getattr(self.dev, "_torn_down", False):
            return
        try:
            opaque.destroy()
        except Exception:
            pass


class SharedWebGpuDevice(Compiled):
    """A tinygrad Compiled device bound to an externally-owned wgpu GPUDevice."""

    def __init__(self, wgpu_device, name: str = DEFAULT_NAME):
        self.wdev = wgpu_device  # the SAME device the renderer uses
        super().__init__(
            name,
            SharedWebGpuAllocator(self),
            [WGSLRenderer],
            functools.partial(SharedWebGPUProgram, self),
        )

    def read_buffer(self, buf) -> bytes:
        """GPU -> host readback through a temporary MAP_READ staging buffer."""
        d = self.wdev
        size = buf.size
        staging = d.create_buffer(size=size, usage=BUF_COPY_DST | BUF_MAP_READ)
        enc = d.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, staging, 0, size)
        d.queue.submit([enc.finish()])
        staging.map_sync(MAP_READ)
        data = bytes(staging.read_mapped())
        staging.unmap()
        try:
            staging.destroy()
        except Exception:
            pass
        return data

    def synchronize(self):
        """Best-effort 'wait for queued work'."""
        for fn in ("_poll", "poll"):
            f = getattr(self.wdev, fn, None)
            if callable(f):
                try:
                    f()
                    return
                except Exception:
                    pass


# --------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------
# Module-level record of what is currently installed, so install() is idempotent. Calling
# it twice with the same device returns the existing instance instead of
# building a second backend on top of the first (which would leave two
# allocators fighting over the same device).
_INSTALLED: dict[str, SharedWebGpuDevice] = {}


def installed(name: str = DEFAULT_NAME) -> SharedWebGpuDevice | None:
    """Return the device installed under `name`, or None."""
    return _INSTALLED.get(name)


def install(
    wgpu_device=None, name: str = DEFAULT_NAME, set_default: bool = True
) -> SharedWebGpuDevice:
    """Make tinygrad's Device[name] resolve to a SharedWebGpuDevice.

    After this, `Tensor(..., device=name)` and `.to(name)` run on the same
    wgpu device the renderer draws from.

    set_default=True also makes `name` the process default device.

    Idempotent: a second call with the same device returns the existing backend.
    """
    if wgpu_device is None:
        wgpu_device = get_shared().device
    existing = _INSTALLED.get(name)
    if existing is not None:
        if existing.wdev is not wgpu_device:
            raise RuntimeError(
                f"device {name!r} is already installed on a different wgpu device; "
                "only one shared device per name is supported"
            )
        return existing

    dev = SharedWebGpuDevice(wgpu_device, name)

    # Inject into tinygrad's device cache so the importlib name lookup (which
    # would find ops_webgpu and build its own Dawn device) is bypassed entirely.
    # This reaches into a private attribute, so it is guarded: if a tinygrad
    # release changes the internals, will give clear error of failure
    getter = getattr(Device, "_Device__get_canonicalized_item", None)
    orig = getattr(getter, "__wrapped__", None)
    if getter is None or orig is None:
        raise RuntimeError(
            "this tinygrad version does not expose "
            "Device._Device__get_canonicalized_item.__wrapped__; "
            "mechiviz.gpu.device.install() needs updating for it"
        )

    getter.cache_clear()

    def patched(ix, _orig=orig, _dev=dev, _name=name):
        # Device strings can carry an index suffix ("WEBGPU:0"), so match on the
        # part before the colon and delegate everything else to tinygrad.
        return _dev if ix.split(":")[0] == _name else _orig(Device, ix)

    Device._Device__get_canonicalized_item = patched  # type: ignore[attr-defined]
    Device._opened_devices.add(name)

    if set_default:
        DEV.value = name

    _INSTALLED[name] = dev
    return dev
