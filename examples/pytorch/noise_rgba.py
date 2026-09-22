import mechiviz as mv
import fastplotlib as fpl
import wgpu
import torch

SIZE = 64
CHANNELS = 3


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

# ------------ plotting

figure = fpl.Figure(size=(600, 600))
figure.canvas.set_title("Noise Demo")
figure[0, 0].axes.visible = False
figure[0, 0].tooltip.enabled = False

image_graphic = figure[0, 0].add_image(
    data=model.state.cpu().numpy(),
    vmin=0,
    vmax=1,
    texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
)

# create a texture
tex = mv.TorchTensorTexture(shape=model.state.shape)
# link the image_graphic texture to the shared texture
tex.texture = image_graphic.data.buffer[0, 0]


label = figure[0, 0].add_text(
    text=f"random {CHANNELS}-channel noise, torch -> shared wgpu memory",
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
