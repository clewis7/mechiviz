import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop
from tinygrad import Tensor

from branchpoint import gpu

SIZE = 64  # 64 f32 per row = 256 bytes -> already row-aligned, no padding
SCALE = 7.0  # 64 * 7 = 448 px on screen

canvas = RenderCanvas(size=(560, 620), title="simple demo")
renderer = gfx.renderers.WgpuRenderer(canvas)

dev = gpu.install()
print(f"installed shared device")


class NoiseModel:
    """One parameter, no gradients, no loss. Just something that changes."""

    def __init__(self, size: int = SIZE):
        self.size = size
        self.state = Tensor.rand(size, size).realize()
        self.step = 0

    def train_step(self):
        self.state.assign(Tensor.rand(self.size, self.size)).realize()
        self.step += 1


model = NoiseModel()

# ---------------------------------------------------------------- 3. scene
scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial("#141414")))

# create a texture
tex = gpu.TinygradTensorTexture(SIZE, SIZE)

# render texture as image in the scene
scene.add(tex.as_image(position=(56, 90, 0), scale=SCALE))

label = gfx.Text(
    text="random noise, tinygrad -> wgpu",
    font_size=15,
    screen_space=False,
    anchor="bottom-center",
    material=gfx.TextMaterial(color="#8fa6b8"),
)
label.local.position = (280, 50, 1)
scene.add(label)

camera = gfx.OrthographicCamera(560, 620)
camera.local.position = (280, 310, 0)


# ---------------------------------------------------------------- 4. loop
def animate():
    if model.step > 300:
        return

    model.train_step()
    tex.update(model.state)
    if model.step % 60 == 0:
        print(f"step {model.step}")
    renderer.render(scene, camera)
    canvas.request_draw()


canvas.request_draw(animate)

if __name__ == "__main__":
    loop.run()
