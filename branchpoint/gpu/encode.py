"""Encode a pygfx texture with the GPU's Vulkan video encoder.

The encoder runs on its own Vulkan device, since wgpu neither enables the video
extensions nor exposes a video-encode queue. Only one resource is shared: a
block of device memory holding the NV12 source picture, allocated through the
patched wgpu-native and imported on the encoder side from a POSIX handle.

Flow per frame, with no host copy of pixel data:

    pygfx renders --> gfx.Texture --compute--> NV12 in shared memory
                  --copy--> NV12 image --encode--> bitstream (host visible)

The only host synchronisation is a wait for the wgpu queue: the two devices
share no timeline, so the conversion must be known complete before the encoder
reads the memory.
"""

import ctypes
import os
import subprocess
from pathlib import Path

import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object

from branchpoint import _native
from .nv12 import Nv12Converter

_SOURCE_DIR = Path(__file__).parent / "_vk_encode"
_LIBRARY = _SOURCE_DIR / "libbp_vk_encode.so"

#: What wgpu validates the buffer against. The VkBuffer underneath gets a
#: fixed usage superset from wgpu-native regardless, which is what the encoder
#: has to reproduce on the import side.
BUFFER_USAGE = _native.USAGE_STORAGE | _native.USAGE_COPY_SRC


class _EncoderInfo(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("coded_width", ctypes.c_uint32),
        ("coded_height", ctypes.c_uint32),
        ("fps_num", ctypes.c_uint32),
        ("fps_den", ctypes.c_uint32),
        ("qp", ctypes.c_uint32),
        ("vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("nv12_fd", ctypes.c_int32),
        ("nv12_mem_size", ctypes.c_uint64),
    ]


def _load_library() -> ctypes.CDLL:
    """dlopen libbp_vk_encode.so, declaring the signatures we call."""
    if not _LIBRARY.exists():
        raise RuntimeError(
            f"{_LIBRARY.name} has not been built. Run `make` in {_SOURCE_DIR} "
            f"(needs Vulkan headers, e.g. the libvulkan-dev package)."
        )
    lib = ctypes.CDLL(str(_LIBRARY))

    lib.bp_encoder_create.restype = ctypes.c_void_p
    lib.bp_encoder_create.argtypes = [
        ctypes.POINTER(_EncoderInfo),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.bp_encoder_destroy.restype = None
    lib.bp_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.bp_encoder_headers.restype = None
    lib.bp_encoder_headers.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.bp_encoder_encode.restype = ctypes.c_int
    lib.bp_encoder_encode.argtypes = lib.bp_encoder_headers.argtypes
    lib.bp_encoder_error.restype = ctypes.c_char_p
    lib.bp_encoder_error.argtypes = [ctypes.c_void_p]
    lib.bp_encoder_info.restype = ctypes.c_char_p
    lib.bp_encoder_info.argtypes = [ctypes.c_void_p]
    return lib


def build(vulkan_include: str | None = None) -> None:
    """Build libbp_vk_encode.so, for when `make` is inconvenient to run."""
    env = dict(os.environ)
    if vulkan_include:
        env["VULKAN_INCLUDE"] = vulkan_include
    subprocess.run(["make", "-C", str(_SOURCE_DIR)], check=True, env=env)


def select_nvidia_adapter():
    """Point pygfx at the NVIDIA GPU, and return the adapter it will use.

    Call this before creating any renderer. On a host with more than one GPU
    wgpu may pick a card that has no Vulkan video encode, and the encoder can
    only import memory exported by the device it runs on. Matching on the name
    is a stopgap: the real test is whether a device reports
    VK_KHR_video_encode_h264, which only the encoder can ask.
    """
    adapters = [
        a
        for a in wgpu.gpu.enumerate_adapters_sync()
        if "nvidia" in a.info["device"].lower()
        and a.info["adapter_type"] == "DiscreteGPU"
    ]
    if not adapters:
        raise RuntimeError(
            "no NVIDIA adapter found among "
            + ", ".join(a.summary for a in wgpu.gpu.enumerate_adapters_sync())
        )
    gfx.renderers.wgpu.select_adapter(adapters[0])
    return adapters[0]


def round_up_16(x: int) -> int:
    """Smallest multiple of 16 that is >= x, the H.264 macroblock grid."""
    return (x + 15) // 16 * 16


class VulkanH264Encoder:
    """All-intra H.264 encoder fed directly from a texture on the GPU.

    Every frame is an IDR at a constant QP, which is the least machinery the
    codec allows: no reference lists, no rate control, one slice per picture.

    Parameters
    ----------
    texture : gfx.Texture or wgpu.GPUTexture
        The picture source. Needs TEXTURE_BINDING usage, which a pygfx render
        target already has. A `gfx.Texture` must have been through one render
        pass so that the wgpu texture behind it exists.
    fps : int
        Written into the SPS timing info. Does not pace anything.
    qp : int
        Quantisation parameter for every slice, 0 (best) to 51.
    device : wgpu.GPUDevice, optional
        Defaults to the device pygfx renders with.

    Attributes
    ----------
    headers : bytes
        SPS and PPS as Annex B NAL units. Write these once at the start of a
        file, or before every frame when streaming to a decoder that may join
        late.
    """

    def __init__(self, texture, fps: int = 30, qp: int = 26, device=None):
        if isinstance(texture, gfx.Texture):
            wgpu_texture = ensure_wgpu_object(texture)
            if wgpu_texture is None:
                raise RuntimeError(
                    "pygfx has not created the GPU texture yet. Render one "
                    "frame into it before creating the encoder."
                )
        else:
            wgpu_texture = texture

        if device is None:
            device = gfx.renderers.wgpu.get_shared().device
        if not (wgpu_texture.usage & wgpu.TextureUsage.TEXTURE_BINDING):
            raise ValueError(
                "texture lacks TEXTURE_BINDING usage, so the conversion pass "
                "cannot read it"
            )

        self.device = device
        width, height = wgpu_texture.size[:2]
        self.size = (width, height)
        self.coded_size = (round_up_16(width), round_up_16(height))
        coded_w, coded_h = self.coded_size

        self._lib = _load_library()
        self._handle = _native.create_exportable_buffer(
            device, coded_w * coded_h * 3 // 2, BUFFER_USAGE
        )
        self._buffer = _native.wrap_as_gpubuffer(self._handle, device, "branchpoint-nv12")
        self._converter = Nv12Converter(
            device, wgpu_texture, self.coded_size, self._buffer
        )

        # The encoder has to run on the same physical device that exported the
        # memory, which is not necessarily the first one Vulkan offers that can
        # encode H.264.
        adapter = device.adapter.info
        info = _EncoderInfo(
            width=width,
            height=height,
            coded_width=coded_w,
            coded_height=coded_h,
            fps_num=fps,
            fps_den=1,
            qp=qp,
            vendor_id=adapter["vendor_id"],
            device_id=adapter["device_id"],
            nv12_fd=self._handle.fd,
            nv12_mem_size=self._handle.alloc_size,
        )
        err = ctypes.create_string_buffer(512)
        self._encoder = self._lib.bp_encoder_create(
            ctypes.byref(info), err, len(err)
        )
        if not self._encoder:
            self._release_buffer()
            raise RuntimeError(f"Vulkan encoder: {err.value.decode()}")
        # The import consumed the handle's file descriptor.
        self._handle.fd = -1

        self._data = ctypes.POINTER(ctypes.c_uint8)()
        self._size = ctypes.c_size_t()
        self._lib.bp_encoder_headers(
            self._encoder, ctypes.byref(self._data), ctypes.byref(self._size)
        )
        self.headers = ctypes.string_at(self._data, self._size.value)
        self.info = self._lib.bp_encoder_info(self._encoder).decode()

    def encode(self) -> bytes:
        """Encode whatever the texture holds now, as one Annex B access unit."""
        encoder = self.device.create_command_encoder()
        self._converter.record(encoder)
        self.device.queue.submit([encoder.finish()])
        # The encoder's device has no view of wgpu's timeline, so the
        # conversion has to be complete before it reads the shared memory.
        self.device._poll_wait()

        if self._lib.bp_encoder_encode(
            self._encoder, ctypes.byref(self._data), ctypes.byref(self._size)
        ) < 0:
            raise RuntimeError(
                f"Vulkan encoder: {self._lib.bp_encoder_error(self._encoder).decode()}"
            )
        return ctypes.string_at(self._data, self._size.value)

    def close(self) -> None:
        """Tear down in reverse ownership order: import, then the export."""
        if getattr(self, "_encoder", None):
            self._lib.bp_encoder_destroy(self._encoder)
            self._encoder = None
        self._release_buffer()

    def _release_buffer(self) -> None:
        if getattr(self, "_buffer", None) is not None:
            self._buffer.destroy()
            self._buffer = None
            for _ in range(3):
                self.device._poll()
        if getattr(self, "_handle", None) is not None:
            _native.free_exportable_memory(self._handle)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
