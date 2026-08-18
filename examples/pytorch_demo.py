"""Torch version of the noise demo: torch -> shared wgpu buffer -> texture -> pygfx."""

import branchpoint as bp  # noqa: F401
import pygfx as gfx
import torch
from rendercanvas.auto import RenderCanvas, loop

SIZE = 64  # 64 f32 per row = 256 bytes -> already row-aligned, no padding
SCALE = 7.0

canvas = RenderCanvas(size=(560, 620), title="torch shared-memory demo")
renderer = gfx.renderers.WgpuRenderer(canvas)

# pygfx's shared wgpu device -- the one all rendering uses
device = gfx.renderers.wgpu.get_shared().device


class NoiseModel:
    """One parameter, no gradients, no loss. Just something that changes."""

    def __init__(self, size: int = SIZE):
        self.size = size
        self.state = torch.rand(size, size, device="cuda")
        self.step = 0

    def train_step(self):
        self.state.uniform_()  # in-place refresh, stays on GPU
        self.step += 1


model = NoiseModel()

# ---------------------------------------------------------------- scene
scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial("#141414")))

tex = bp.gpu.TorchTensorTexture(device=device, width=SIZE, height=SIZE)


scene.add(tex.as_image(position=(56, 90, 0), scale=SCALE))
# -------------------------------------------------------------------------

label = gfx.Text(
    text="random noise, torch -> shared wgpu memory",
    font_size=15,
    screen_space=False,
    anchor="bottom-center",
    material=gfx.TextMaterial(color="#8fa6b8"),
)
label.local.position = (280, 50, 1)
scene.add(label)

camera = gfx.OrthographicCamera(560, 620)
camera.local.position = (280, 310, 0)


# ---------------------------------------------------------------- loop
def animate():
    if model.step > 300:
        return
    model.train_step()
    tex.update(model.state)  # one D2D copy + one buffer->texture blit
    if model.step % 60 == 0:
        print(f"step {model.step}")
    renderer.render(scene, camera)
    canvas.request_draw()


canvas.request_draw(animate)

if __name__ == "__main__":
    loop.run()
