"""A rendercanvas backend that streams H.264 instead of downloading pixels.

rendercanvas' http backend renders into a texture, downloads that texture to
host RAM, and JPEG-encodes it on the CPU. The download is 14.7 MB a frame at
1440p and 33.2 MB at 4K, which is 2 GB/s over the bus at 60 fps.

Here the rendered pixels never leave the GPU. pygfx renders into the context's
texture, the Vulkan encoder reads that texture, and the first bytes to reach
host RAM are already H.264 -- a few hundred kB instead of tens of MB, and the
same bytes go straight out over the websocket.

Everything else is rendercanvas' http backend unchanged: the ASGI app, the
websocket, the frame flow control, and mouse and keyboard events.

    from branchpoint.gpu.h264_canvas import RenderCanvas, loop

    canvas = RenderCanvas(size=(1280, 720))
    ...
    loop.run()

Every picture is an IDR and the parameter sets are repeated ahead of each one,
so a browser can connect at any time and the flow control is free to drop
frames without the decoder losing state.
"""

import logging
import time
from pathlib import Path

import wgpu
from rendercanvas import http as _http
from rendercanvas.contexts.wgpucontext import WgpuContextToBitmap

from .encode import VulkanH264Encoder

logger = logging.getLogger(__name__)

__all__ = ["H264RenderCanvas", "RenderCanvas", "asgi", "loop"]

#: Served to the browser in place of rendercanvas' own client, which displays
#: frames in an <img>. This one decodes into a <canvas> with WebCodecs.
_CLIENT_JS = (Path(__file__).parent / "_h264_client.js").read_text()

_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>RenderCanvas over http, H.264</title>
    <script type='module' src='renderview.js'></script>
    <script type='module' src='renderview-h264.js'></script>
    <link rel="stylesheet" href="renderview.css">
</head>
<body>
  <h1>RenderCanvas over http, H.264</h1>

    <div id='canvas' class='renderview-wrapper is-resizable' style='width:640px; height:480px'>
        <p style='width:100%; height:100%; background:#aaa; display: flex; justify-content: center; align-items: center; font-size:150%'>Loading ...</p>
    </div>

    <div id='status' style='position:fixed; top:0; right:0; background:#ccc; color:#000; padding:1em; font-family: monospace'></div>
</body>
</html>
"""

# The http backend serves from one module-level dict, and assumes one canvas
# per process, so swapping the page and adding our client is the way in.
_http.resources["renderview-h264.js"] = ("text/javascript", _CLIENT_JS)
_http.resources["index.html"] = ("text/html", _HTML)

asgi = _http.asgi
loop = _http.loop


def codec_string(headers: bytes) -> str:
    """The avc1 codec string WebCodecs needs, read out of the SPS.

    profile_idc, the constraint flags and level_idc are the three bytes that
    follow the SPS NAL header, and they are exactly what the codec string
    spells out, so read them rather than guess a level.
    """
    for i in range(len(headers) - 6):
        if headers[i : i + 3] == b"\x00\x00\x01" and headers[i + 3] & 0x1F == 7:
            profile, constraints, level = headers[i + 4 : i + 7]
            return f"avc1.{profile:02x}{constraints:02x}{level:02x}"
    raise RuntimeError("the encoder's parameter sets contain no SPS")


class H264WgpuContext(WgpuContextToBitmap):
    """A wgpu context that encodes its texture instead of downloading it.

    The base class already keeps a single texture and only replaces it when the
    canvas resizes, which is what lets the encoder bind it once.
    """

    def __init__(self, present_info: dict):
        super().__init__(present_info)
        self._encoder = None
        self._encoded_texture = None
        self._codec = None
        self._unusable_texture = None
        settings = present_info.get("h264", {})
        self._qp = settings.get("qp", 30)
        self._fps = settings.get("fps", 60)

    def _get_capabilities(self):
        capabilities = super()._get_capabilities()
        # The conversion pass samples the texture. The base class only asks for
        # COPY_SRC, which is all its download needs.
        self._context_texture_usage |= wgpu.TextureUsage.TEXTURE_BINDING
        return capabilities

    def _rc_present(self, *, force_sync: bool = False) -> dict:
        texture = self._texture
        if texture is None:
            return {"method": "skip"}

        if texture is self._unusable_texture:
            return {"method": "skip"}

        if texture is not self._encoded_texture:
            # First frame, or the canvas resized and the base class made a new
            # texture. Either way the encoder is bound to the old one.
            self._close_encoder()
            try:
                self._encoder = VulkanH264Encoder(
                    texture, fps=self._fps, qp=self._qp, device=self._config["device"]
                )
            except RuntimeError as err:
                # A canvas starts out at a placeholder size and only learns its
                # real one when a client reports it, and that placeholder is
                # below what the encoder will take. Skip until the size changes.
                self._unusable_texture = texture
                logger.info("not encoding a %sx%s frame: %s", *texture.size[:2], err)
                return {"method": "skip"}
            self._encoded_texture = texture
            self._codec = codec_string(self._encoder.headers)

        width, height = texture.size[:2]
        return {
            "method": "h264",
            "data": self._encoder.headers + self._encoder.encode(),
            "width": width,
            "height": height,
            "codec": self._codec,
        }

    def _close_encoder(self):
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
        self._encoded_texture = None
        self._codec = None

    def _drop_texture(self):
        self._close_encoder()
        super()._drop_texture()

    def _rc_close(self):
        self._close_encoder()
        super()._rc_close()


class H264RenderCanvas(_http.HttpRenderCanvas):
    """An http canvas whose frames are H.264 encoded on the GPU.

    Parameters
    ----------
    qp : int
        Quantisation parameter for every slice, 0 (best) to 51. Trades quality
        against bytes on the wire; the frames are all-intra, so this is the
        only quality knob.
    fps : int
        Written into the stream's timing info. Does not pace anything.
    """

    def __init__(self, *args, qp: int = 30, fps: int = 60, **kwargs):
        self._qp = int(qp)
        self._fps = int(fps)
        size = kwargs.get("size") or (640, 480)
        super().__init__(*args, **kwargs)
        # An http canvas has a physical size of 1x1 until a client reports its
        # own, and drawing at that size makes a mess of anything that reserves
        # margins for a title or toolbar. Start at the requested size instead;
        # a client's resize event corrects it, as it would anyway.
        self._size_info.set_physical_size(
            max(1, round(size[0])), max(1, round(size[1])), 1
        )

    def get_context(self, context_type="wgpu"):
        """Hand pygfx the encoding context when it asks for a wgpu one."""
        if context_type == "wgpu":
            context_type = H264WgpuContext
        return super().get_context(context_type)

    def _rc_get_present_info(self, present_methods):
        # rgba-u8 resolves to an -srgb texture format, which is what the
        # conversion pass expects to read linear light from.
        if "bitmap" in present_methods:
            return {
                "method": "bitmap",
                "formats": ["rgba-u8"],
                "h264": {"qp": self._qp, "fps": self._fps},
            }
        else:
            return None  # raises error

    def _rc_present_h264(self, *, data, width, height, codec):
        """Store the encoded frame and send it on to whoever is ready for it."""
        timestamp = time.time()
        self._stats["encoded_frames"] += 1
        if self._stats["start_time"] <= 0:
            self._stats["start_time"] = timestamp

        msg = dict(
            type="framebufferdata",
            nbuffers=1,
            mimetype="video/h264",
            codec=codec,
            width=width,
            height=height,
            timestamp=timestamp,
            index=0,
        )
        self._last_frame = msg, [data]
        self._ref_index += 1
        self._send_last_frame_to_ready_clients()


RenderCanvas = H264RenderCanvas
