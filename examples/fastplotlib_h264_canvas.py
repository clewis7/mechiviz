"""Serve an interactive fastplotlib figure to a browser as H.264.

    python examples/fastplotlib_h264_canvas.py

then open http://localhost:60649. The client decodes with WebCodecs, so it
needs Chrome/Edge 94+, Safari 16.4+ or Firefox 130+.

The rendered pixels never leave the GPU. pygfx draws into the canvas' texture,
the Vulkan encoder reads that texture, and the only bytes crossing the bus are
already H.264 -- where rendercanvas' own http backend would download the whole
frame and JPEG it on the CPU.

Mouse and keyboard events come back over the same websocket as always, so the
figure pans, zooms and picks the way it does in a desktop window.
"""

from branchpoint.gpu.encode import select_nvidia_adapter
from branchpoint.gpu.h264_canvas import RenderCanvas, loop

# branchpoint has to be imported before anything that pulls in wgpu; importing
# it is what points wgpu-native at the build with the exportable-buffer entry
# points, and that only works before the library is dlopened.
import numpy as np
import fastplotlib as fpl
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui
import imageio.v3 as iio
from skimage.filters import gaussian

select_nvidia_adapter()

figure = fpl.Figure(cameras="3d", controller_types="orbit", canvas=RenderCanvas(size=(1920, 1080), qp=32, max_fps=60))

data = iio.imread("imageio:stent.npz")
# MIP rendering is the default `mode`
vol_mip = figure[0, 0].add_image_volume(gaussian(data, sigma=2.0))

# make another graphic to show a slice of the volume
vol_slice = figure[0, 0].add_image_volume(
    vol_mip.data,  # pass the data property from the previous volume so they share the same buffer on the GPU
    mode="slice",
    plane=(0, -0.5, -0.5, 50),
    offset=(150, 0, 0)  # place the graphic at x=150
)

class GUI(ImguiWindow):
    def __init__(self):
        super().__init__()
        self._sigma = 2

    def update(self):
        changed, self._sigma = imgui.slider_int("sigma", v=self._sigma, v_min=0, v_max=5)

        if changed:
            vol_mip.data = gaussian(data, sigma=self._sigma)
            vol_mip.reset_vmin_vmax()
            vol_slice.reset_vmin_vmax()

        imgui.text("Select plane defined by:\nax + by + cz + d = 0")
        _, a = imgui.slider_float("a", v=vol_slice.plane[0], v_min=-1, v_max=1.0)
        _, b = imgui.slider_float("b", v=vol_slice.plane[1], v_min=-1, v_max=1.0)
        _, c = imgui.slider_float("c", v=vol_slice.plane[2], v_min=-1, v_max=1.0)

        largest_dim = max(vol_slice.data.value.shape)
        _, d = imgui.slider_float(
            "d", v=vol_slice.plane[3], v_min=0, v_max=largest_dim * 2
        )

        vol_slice.plane = (a, b, c, d)

gui = GUI()
figure.add_imgui_window(gui, location="right", size=200, title="change data buffer")
figure.show()

# Nothing renders until a browser connects; the client's frame feedback is what
# paces the loop.
if __name__ == "__main__":
    loop.run()
