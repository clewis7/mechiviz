import mechiviz as mechi
import pygfx as gfx
import torch
from rendercanvas.auto import RenderCanvas, loop

SIZE = 64
CHANNELS = 3
SCALE = 7.0

canvas = RenderCanvas(size=(560, 620), title="torch shared-memory demo (rgb)")
renderer = gfx.renderers.WgpuRenderer(canvas)

# pygfx's shared wgpu device -- the one all rendering uses
device = gfx.renderers.wgpu.get_shared().device


class NoiseModel:
    """One parameter, no gradients, no loss. Just something that changes."""

    def __init__(self, size: int = SIZE, channels: int = CHANNELS):
        self.state = torch.rand(size, size, channels, device="cuda")
        self.step = 0

    def train_step(self):
        # Smooth drift rather than pure noise
        self.state.add_(torch.randn_like(self.state) * 0.05).clamp_(0, 1)
        self.step += 1


model = NoiseModel()

# ---------------------------------------------------------------- scene
scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial("#141414")))

tex = mechi.TorchTensorTexture(shape=model.state.shape, device=device)
print(
    f"source {tuple(model.state.shape)} -> texture format {tex.format} "
    f"({tex.n_channels} channels stored)"
)

scene.add(tex.as_image(position=(56, 90, 0), scale=SCALE))

label = gfx.Text(
    text=f"random {CHANNELS}-channel noise, torch -> shared wgpu memory",
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
    tex.update(model.state)  # one D2D copy (3 of 4 channels) + one blit
    if model.step % 60 == 0:
        print(f"step {model.step}")
    renderer.render(scene, camera)
    canvas.request_draw()


canvas.request_draw(animate)

if __name__ == "__main__":
    loop.run()
