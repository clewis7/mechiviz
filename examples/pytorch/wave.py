import mechiviz as mv
import fastplotlib as fpl
import torch

N_LINES = 16
N_POINTS = 256


class WaveModel:
    def __init__(self, n_lines: int = N_LINES, n_points: int = N_POINTS):
        self.n_lines = n_lines
        self.n_points = n_points
        self.step = 0

        x = torch.linspace(0, 4 * torch.pi, n_points, device="cuda")
        self.x = x[None].expand(n_lines, -1)
        self.freq = torch.linspace(0.5, 2.0, n_lines, device="cuda")[:, None]
        self.phase = torch.zeros(n_lines, 1, device="cuda")

        self.positions = torch.zeros(n_lines, n_points, 3, device="cuda")
        self._write()

    def _write(self):
        self.positions[..., 0] = self.x
        self.positions[..., 1] = torch.sin(self.x * self.freq + self.phase)

    def train_step(self):
        self.phase += 0.05  # in-place, stays on GPU
        self._write()
        self.step += 1


model = WaveModel()

# -------------- plotting

figure = fpl.Figure(size=(900, 700))
figure.canvas.set_title("Line Demo")
figure[0, 0].axes.visible = False
figure[0, 0].tooltip.enabled = False

lc = figure[0, 0].add_line_stack(
    data=model.positions.cpu().numpy(),
    cmap="tab20",
    thickness=2.0,
)

# create a buffer per line
bufs = [mv.TorchTensorBuffer(shape=g.data.value.shape) for g in lc.graphics]
# link each line graphic's positions to its shared buffer
for buf, g in zip(bufs, lc.graphics):
    buf.buffer = g.data.buffer

label = figure[0, 0].add_text(
    text="waves, pytorch -> wgpu",
    font_size=15,
    anchor="bottom-left",
    offset=(0, -3, 0),
)

figure.show()
figure[0, 0].auto_scale(maintain_aspect=False)


# ------------- update
def animate():
    if model.step > 600:
        return

    model.train_step()
    for i, buf in enumerate(bufs):
        buf.update(model.positions[i], synchronize=(i == 0))
    if model.step % 60 == 0:
        print(f"step {model.step}")


figure[0, 0].add_animations(animate)

if __name__ == "__main__":
    fpl.loop.run()
