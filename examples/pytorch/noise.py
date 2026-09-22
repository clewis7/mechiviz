"""Torch version of the noise demo: torch -> shared wgpu buffer -> texture -> pygfx."""

import mechiviz as mv
import fastplotlib as fpl
import wgpu
import torch

SIZE = 64  # 64 f32 per row = 256 bytes -> already row-aligned, no padding


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

# ------------ plotting

figure = fpl.Figure(size=(600, 600))
figure.canvas.set_title("Noise Demo")
figure[0, 0].axes.visible = False
figure[0, 0].tooltip.enabled = False

image_graphic = figure[0, 0].add_image(
    data=model.state.cpu().numpy(),
    cmap="gray",
    vmin=0,
    vmax=1,
    texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
)

# create a texture
tex = mv.TorchTensorTexture(shape=model.state.shape)
# link the image_graphic texture to the shared texture
tex.texture = image_graphic.data.buffer[0, 0]


label = figure[0, 0].add_text(
    text="random noise, pytorch -> wgpu",
    font_size=15,
    anchor="bottom-center",
    offset=(int(SIZE / 2), SIZE + 5, 0),
)

figure.show()


# ------------- update
def animate():
    if model.step > 300:
        return

    model.train_step()
    tex.update(model.state)
    if model.step % 60 == 0:
        print(f"step {model.step}")


figure[0, 0].add_animations(animate)

if __name__ == "__main__":
    fpl.loop.run()
