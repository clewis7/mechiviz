import math

import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
import numpy as np
from tinygrad import Tensor, nn
import pygfx as gfx
import wgpu

from branchpoint import gpu
from induction_model import Transformer

# create the shared device
dev = gpu.install()


TILE = 64  # = seq len; 64*4 = 256 bytes/row -> aligned
VOCAB, SEQ, HALF = 64, 64, 32
BATCH, LR = 32, 3e-3
N_HEADS = 4
PROBE_BS = 4  # fixed probe batch for all spatial views
STEPS_PER_FRAME = 1  # raise to train faster than you render
MAX_PTS = 4000  # points kept in the line plots
MAX_STEP = 300
HEAD_COLORS = ["#66d9ff", "#7dff9e", "#ffb066", "#ff7de1"]
ABLATE = [False] * N_HEADS


# create figure
figure = fpl.Figure(size=(1180, 760), names=[" "])
figure.canvas.set_title("Synthetic Induction Task")
figure[0, 0].axes.visible = False

# ---------------- 3. data: induction batches + fixed probe ----------------
rng = np.random.default_rng(0)


def make_batch(bs: int) -> Tensor:
    first = rng.integers(0, VOCAB, size=(bs, HALF))
    return Tensor(np.concatenate([first, first], axis=1).astype(np.int32))


# fixed probe batch: constant across training so evolution is visible, not flicker
_pf = np.random.default_rng(123).integers(0, VOCAB, size=(PROBE_BS, HALF))
PROBE = Tensor(np.concatenate([_pf, _pf], axis=1).astype(np.int32)).realize()

# induction-stripe mask: ones at (i, i-HALF+1) for i in the repeat region.
# (pattern * MASK).sum(-1,-2) / HALF = mean attention mass on the stripe.
_m = np.zeros((SEQ, SEQ), np.float32)
for i in range(HALF, SEQ):
    _m[i, i - HALF + 1] = 1.0
STRIPE_MASK = Tensor(_m).realize()

# ---------------- 4. model + training step ----------------
# constructed AFTER install_shared_webgpu(set_default=True): weights, causal
model = Transformer(vocab=VOCAB, seq_len=SEQ, n_heads=N_HEADS)
opt = nn.optim.Adam(model.parameters(), lr=LR)


def head_mask():
    return Tensor([0.0 if a else 1.0 for a in ABLATE]).realize()


def train_step() -> Tensor:
    with Tensor.train():
        opt.zero_grad()
        tokens = make_batch(BATCH)
        preds = model(tokens, head_mask())[:, HALF - 1 : SEQ - 1]  # predict the repeat
        loss = (
            preds.reshape(-1, VOCAB)
            .sparse_categorical_crossentropy(tokens[:, HALF:SEQ].reshape(-1))
            .backward()
        )
        opt.step()
    return loss


# ---------------- 5. scene ----------------
scene = figure[0, 0].scene


def make_label(text, pos, size=14, color="#dddddd", anchor="middle-center"):
    try:
        t = gfx.Text(
            text=str(text),
            font_size=size,
            screen_space=False,
            anchor=anchor,
            material=gfx.TextMaterial(color=color),
        )
        t.local.position = pos
        scene.add(t)
        return t
    except Exception as e:
        print(f"(label '{text}' skipped: {type(e).__name__})")
        return None


def add_tile(cx, cy, sx, sy):
    """Same texture/image setup as cnn_mnist_grid.add_tile, with free scale.
    (cx, cy) is the tile's lower-left corner; negative y-scale flips the image
    so row 0 (dst position 0) is at the top, as attention plots are usually read."""
    tex = gfx.Texture(
        size=(TILE, TILE, 1),
        dim=2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    img = gfx.Image(gfx.Geometry(grid=tex), gfx.ImageBasicMaterial(clim=(0, 255)))
    img.local.position = (cx, cy + TILE * sy, 0)
    img.local.scale = (sx, -sy, 1)
    scene.add(img)
    return tex


# -- row of head heatmaps (top) --
HSCALE = 2.6  # 64 px -> ~166 px tiles
HW = TILE * HSCALE
GAP = 26
row_w = N_HEADS * HW + (N_HEADS - 1) * GAP
ROW_X0 = (1180 - row_w) / 2  # left edge of the row
HEAD_Y = 470  # lower-left y of head tiles

head_tiles = []
for h in range(N_HEADS):
    x = ROW_X0 + h * (HW + GAP)
    head_tiles.append(add_tile(x, HEAD_Y, HSCALE, HSCALE))
    make_label(
        f"head {h}",
        (x + HW / 2, HEAD_Y + HW + 8, 1),
        size=15,
        color=HEAD_COLORS[h],
        anchor="bottom-center",
    )
make_label(
    "attention patterns on fixed probe (batch mean, dst row x src col)",
    (590, HEAD_Y + HW + 34, 1),
    size=13,
    color="#888888",
    anchor="bottom-center",
)

# -- per-position loss strip (middle): (1, HALF) tensor upscaled into a wide bar --
STRIP_H = 26
STRIP_Y = HEAD_Y - 74
strip_tex = add_tile(ROW_X0, STRIP_Y, row_w / TILE, STRIP_H / TILE)
make_label(
    "per-position loss on probe (pos 32..63; bright = high)",
    (590, STRIP_Y + STRIP_H + 8, 1),
    size=13,
    color="#888888",
    anchor="bottom-center",
)

# -- plot area (bottom): loss (log) + stripe metric per head --
PLOT_X0, PLOT_X1 = 120, 1060
PLOT_Y0, PLOT_Y1 = 55, 300
LOG_LO, LOG_HI = -4.5, 0.8  # log10(loss) axis range

scene.add(
    gfx.Line(
        gfx.Geometry(
            positions=np.array(
                [
                    [PLOT_X0, PLOT_Y0, 0],
                    [PLOT_X1, PLOT_Y0, 0],
                    [PLOT_X1, PLOT_Y1, 0],
                    [PLOT_X0, PLOT_Y1, 0],
                    [PLOT_X0, PLOT_Y0, 0],
                ],
                np.float32,
            )
        ),
        gfx.LineMaterial(color="#444444", thickness=1.5),
    )
)
make_label(
    "loss (log scale)",
    (PLOT_X0, PLOT_Y1 + 8, 1),
    size=13,
    color="#ffffff",
    anchor="bottom-left",
)
make_label(
    "induction-stripe attn per head (0..1)",
    (PLOT_X1, PLOT_Y1 + 8, 1),
    size=13,
    color="#aaaaaa",
    anchor="bottom-right",
)


def make_line(color, thickness=2.0):
    pos = np.full((MAX_PTS, 3), np.nan, np.float32)  # NaN points are skipped
    pos[:, 2] = 0.0
    geom = gfx.Geometry(positions=pos)
    scene.add(gfx.Line(geom, gfx.LineMaterial(color=color, thickness=thickness)))
    return geom


loss_geom = make_line("#ffffff", 2.5)
stripe_geoms = [make_line(c, 1.6) for c in HEAD_COLORS]


def update_line(geom, ys, lo, hi):
    """ys: full history (python list). Maps index -> x, value -> y in plot rect.
    If history exceeds MAX_PTS, it's decimated so the whole run stays visible."""
    if len(ys) > MAX_PTS:
        idx = np.linspace(0, len(ys) - 1, MAX_PTS).astype(int)
        ys_a = np.asarray(ys, np.float32)[idx]
    else:
        ys_a = np.asarray(ys, np.float32)
    n = len(ys_a)
    data = geom.positions.data
    if n:
        data[:n, 0] = PLOT_X0 + np.arange(n) / max(n - 1, 1) * (PLOT_X1 - PLOT_X0)
        data[:n, 1] = (
            np.clip((ys_a - lo) / (hi - lo), 0, 1) * (PLOT_Y1 - PLOT_Y0) + PLOT_Y0
        )
        data[n:, 0] = np.nan
    try:
        geom.positions.update_full()
    except AttributeError:  # older pygfx
        geom.positions.update_range(0, MAX_PTS)


camera = figure[0, 0].camera
camera.local.position = (590, 380, 0)


# ---------------- 7. probe pass: spatial views + plot scalars ----------------
def probe_views():
    logits = model(PROBE)  # fills model.acts
    pattern = model.acts["attn_pattern"]  # (B, H, 64, 64)
    pat_mean = pattern.mean(axis=0).realize()  # (H, 64, 64)

    # per-position NLL over the repeat region: -log p(target) at each pos
    lp = logits[:, HALF - 1 : SEQ - 1].log_softmax(-1)  # (B, HALF, V)
    tgt = PROBE[:, HALF:SEQ]  # (B, HALF)
    nll = -(lp * tgt.one_hot(VOCAB)).sum(-1)  # (B, HALF)
    pos_loss = nll.mean(axis=0).reshape(1, HALF).realize()  # (1, HALF)

    # stripe metric per head (host readback: 4 floats)
    stripe = (
        ((pattern * STRIPE_MASK).sum(axis=(-1, -2)) / HALF).mean(axis=0).numpy()
    )  # (H,)
    return pat_mean, pos_loss, stripe


# ---------------- 2. guis ----------------
class MenuGUI(EdgeWindow):
    def __init__(self, figure, size, location, title):
        super().__init__(
            figure=figure,
            size=size,
            location=location,
            title=title,
            window_flags=imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_scrollbar,
        )
        self._step = 0
        self._loss_hist: list[float] = []
        self._stripe_hist: list[list[float]] = [[] for _ in range(N_HEADS)]

        self._paused = True

    def update(self):
        global model, opt
        if imgui.button("Restart"):
            model = Transformer(vocab=VOCAB, seq_len=SEQ, n_heads=N_HEADS)
            opt = nn.optim.Adam(model.parameters(), lr=LR)
            self._step = 0

        imgui.same_line()

        labels = ["Checkpoint", "Load Checkpoint"]

        if self._paused:
            labels.insert(0, "Train")
        else:
            labels.insert(0, "Pause")

        # total width = buttons + spacing between them
        style = imgui.get_style()
        spacing = style.item_spacing.x
        total_width = sum(
            imgui.calc_text_size(l).x + style.frame_padding.x * 2 for l in labels
        ) + spacing * (len(labels) - 1)

        # push cursor to the right edge
        avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - total_width)

        for i, label in enumerate(labels):
            if i > 0:
                imgui.same_line()
            if imgui.button(label):
                if label in ["Pause", "Train"]:
                    self._paused = not self._paused
                    if self._paused:
                        print(f"Training paused, step {self._step}")

        if not self._paused:
            if self._step > MAX_STEP:
                return

            last_loss = None
            for _ in range(STEPS_PER_FRAME):
                last_loss = train_step()
                self._step += 1
            self._loss_hist.append(max(last_loss.item(), 1e-6))  # scalar readback

            pat_mean, pos_loss, stripe = probe_views()
            for h in range(N_HEADS):
                gpu.copy_tensor_to_texture(
                    dev, gpu.pack_rgba8(pat_mean[h]), head_tiles[h]
                )
                self._stripe_hist[h].append(float(stripe[h]))
            gpu.copy_tensor_to_texture(dev, gpu.pack_rgba8(pos_loss), strip_tex)

            update_line(
                loss_geom, [math.log10(v) for v in self._loss_hist], LOG_LO, LOG_HI
            )
            for h in range(N_HEADS):
                update_line(stripe_geoms[h], self._stripe_hist[h], 0.0, 1.0)

            if self._step % 25 < STEPS_PER_FRAME:
                print(
                    f"step {self._step:4d}  loss {self._loss_hist[-1]:.4f}  stripe {np.round(stripe, 2)}"
                )


class EdgeGUI(EdgeWindow):
    def __init__(self, figure, size, location, title):
        super().__init__(
            figure=figure,
            size=size,
            location=location,
            title=title,
            window_flags=imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize,
        )

        self._learning_rate = LR

    def _make_title(self, text: str):
        imgui.separator()
        avail = imgui.get_content_region_avail().x
        text_w = imgui.calc_text_size(text).x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail - text_w) * 0.5)
        imgui.text(text)
        imgui.separator()

    def update(self):
        self._make_title("Learning Rate")

        # learning rate slider
        imgui.text("lr:")
        imgui.same_line()
        changed, lr = imgui.slider_float("##LR", self._learning_rate, 1e-6, 1e-1)

        if changed:
            self._learning_rate = lr
            opt.lr = self._learning_rate

        self._make_title("Head Ablation")

        for i in range(N_HEADS):
            changed, _ = imgui.checkbox(f"h{i}", ABLATE[i])
            if changed:
                ABLATE[i] = _
            if i != N_HEADS - 1:
                imgui.same_line()


# make GUI instance
gui = MenuGUI(
    figure,  # the figure this GUI instance should live inside
    size=30,  # width or height of the GUI window within the figure
    location="top",  # the edge to place this window at
    title=" ",  # window title
)

gui2 = EdgeGUI(figure, size=200, location="right", title="Training Params")

# add guis to the figure
figure.add_gui(gui)
figure.add_gui(gui2)

figure[0, 0].camera.show_object(
    figure[0, 0].scene, view_dir=(0, 0, -1), up=(0, 1, 0), scale=0.7
)
figure[0, 0].controller.enabled = False

figure.show()


# NOTE: fpl.loop.run() should not be used for interactive sessions
# See the "JupyterLab and IPython" section in the user guide
if __name__ == "__main__":
    print(__doc__)
    fpl.loop.run()
