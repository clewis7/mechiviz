import math


import mechiviz as mv
import fastplotlib as fpl
import numpy as np
import pygfx as gfx
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui
from tinygrad import Tensor, nn


from model import Transformer

TILE = 64  # = seq len; 64 float32 = 256 bytes/row, so no row padding is needed
VOCAB, SEQ, HALF = 64, 64, 32
BATCH, LR = 32, 3e-3
N_HEADS = 4
PROBE_BS = 4  # fixed probe batch, so the heatmaps evolve rather than flicker
STEPS_PER_FRAME = 1
MAX_PTS = 4000
MAX_STEP = 300
HEAD_COLORS = ["#66d9ff", "#7dff9e", "#ffb066", "#ff7de1"]
ABLATE = [False] * N_HEADS

# create figure
figure = fpl.Figure(size=(1180, 760), names=[" "])
figure.canvas.set_title("Synthetic Induction Task")
figure[0, 0].axes.visible = False

# install shared device
dev = mv.install()

# create data
rng = np.random.default_rng(0)


def make_batch(bs: int) -> Tensor:
    first = rng.integers(0, VOCAB, size=(bs, HALF))
    return Tensor(np.concatenate([first, first], axis=1).astype(np.int32))


_pf = np.random.default_rng(123).integers(0, VOCAB, size=(PROBE_BS, HALF))
PROBE = Tensor(np.concatenate([_pf, _pf], axis=1).astype(np.int32)).realize()

# induction-stripe mask: ones at (i, i-HALF+1) for i in the repeat region.
_m = np.zeros((SEQ, SEQ), np.float32)
for i in range(HALF, SEQ):
    _m[i, i - HALF + 1] = 1.0
STRIPE_MASK = Tensor(_m).realize()

# model, optimizer, and training step
model = Transformer(vocab=VOCAB, seq_len=SEQ, n_heads=N_HEADS)
opt = nn.optim.Adam(model.parameters(), lr=LR)


def head_mask() -> Tensor:
    """0/1 per head. A mask multiply rather than a Python branch, so ablation
    stays inside the graph and gradients to a dead head go to zero.
    """
    return Tensor([0.0 if a else 1.0 for a in ABLATE]).realize()


def train_step() -> Tensor:
    with Tensor.train():
        opt.zero_grad()
        tokens = make_batch(BATCH)
        preds = model(tokens, head_mask())[:, HALF - 1 : SEQ - 1]
        loss = (
            preds.reshape(-1, VOCAB)
            .sparse_categorical_crossentropy(tokens[:, HALF:SEQ].reshape(-1))
            .backward()
        )
        opt.step()
    return loss


def probe_views():
    """One eval pass on the fixed probe, producing everything the views need."""
    logits = model(PROBE)
    pattern = model.acts["attn_pattern"]  # (B, H, 64, 64)
    pat_mean = pattern.mean(axis=0).realize()  # (H, 64, 64)

    lp = logits[:, HALF - 1 : SEQ - 1].log_softmax(-1)
    tgt = PROBE[:, HALF:SEQ]
    nll = -(lp * tgt.one_hot(VOCAB)).sum(-1)  # (B, HALF)
    pos_loss = nll.mean(axis=0).reshape(1, HALF).realize()  # (1, HALF)

    stripe = ((pattern * STRIPE_MASK).sum(axis=(-1, -2)) / HALF).mean(axis=0).numpy()
    return pat_mean, pos_loss, stripe


# adding viz stuff to the scene
scene = figure[0, 0].scene
camera = figure[0, 0].camera


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


# -- row of head heatmaps --
HSCALE = 2.6
HW = TILE * HSCALE
GAP = 26
row_w = N_HEADS * HW + (N_HEADS - 1) * GAP
ROW_X0 = (1180 - row_w) / 2
HEAD_Y = 470

head_tex = []
for h in range(N_HEADS):
    x = ROW_X0 + h * (HW + GAP)
    # r32float + a FIXED clim: attention weights are already in [0, 1], so
    # brightness stays comparable across frames. Per-frame min/max would make
    # a faint stripe and a sharp one look identical, hiding the thing we came
    # to watch.
    t = mv.TinygradTensorTexture(shape=(TILE, TILE))
    scene.add(t.as_image(clim=(0.0, 1.0), position=(x, HEAD_Y, 0), scale=HSCALE))
    head_tex.append(t)
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

# -- per-position loss strip --
# (1, HALF) = 128 bytes per row, NOT 256-aligned, so this is the one view that
# exercises TensorTexture's padding path. clim upper bound is ln(64) ~= 4.16,
# chance level, so a fully-untrained strip reads as full brightness.
STRIP_H = 26
STRIP_Y = HEAD_Y - 74
strip_tex = mv.TinygradTensorTexture(shape=(1, HALF))
scene.add(
    strip_tex.as_image(
        clim=(0.0, 4.16),
        position=(ROW_X0, STRIP_Y, 0),
        scale=(row_w / HALF, STRIP_H / 1.0),
    )
)
make_label(
    "per-position loss on probe (pos 32..63; bright = high)",
    (590, STRIP_Y + STRIP_H + 18, 1),
    size=13,
    color="#888888",
    anchor="bottom-center",
)

# -- plot area --
PLOT_X0, PLOT_X1 = 120, 1060
PLOT_Y0, PLOT_Y1 = 55, 300
LOG_LO, LOG_HI = -4.5, 0.8

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
    """Plain pygfx line, updated from numpy.

    Not a TensorBuffer: these values come from .item()/.numpy() readbacks we
    already do for the printout, so they are on the host. Uploading them to the
    GPU only to copy them into the vertex buffer would add work, not remove it.
    Use TensorBuffer when a metric is computed on the GPU and never needed on
    the host.
    """
    pos = np.full((MAX_PTS, 3), np.nan, np.float32)  # NaN points are skipped
    pos[:, 2] = 0.0
    geom = gfx.Geometry(positions=pos)
    scene.add(gfx.Line(geom, gfx.LineMaterial(color=color, thickness=thickness)))
    return geom


loss_geom = make_line("#ffffff", 2.5)
stripe_geoms = [make_line(c, 1.6) for c in HEAD_COLORS]


def update_line(geom, ys, lo, hi):
    """Map index -> x and value -> y into the plot rect. Decimates past MAX_PTS
    so the whole run stays visible rather than scrolling off.
    """
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


camera.local.position = (590, 380, 0)


# guis
class MenuGUI(ImguiWindow):
    def __init__(self):
        super().__init__()
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
            self._loss_hist.clear()
            for hist in self._stripe_hist:
                hist.clear()

        imgui.same_line()

        labels = ["Checkpoint", "Load Checkpoint"]
        labels.insert(0, "Train" if self._paused else "Pause")

        style = imgui.get_style()
        spacing = style.item_spacing.x
        total_width = sum(
            imgui.calc_text_size(l).x + style.frame_padding.x * 2 for l in labels
        ) + spacing * (len(labels) - 1)
        avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - total_width)

        for i, label in enumerate(labels):
            if i > 0:
                imgui.same_line()
            if imgui.button(label) and label in ("Pause", "Train"):
                self._paused = not self._paused
                if self._paused:
                    print(f"Training paused, step {self._step}")

        if self._paused or self._step > MAX_STEP:
            return

        last_loss = None
        for _ in range(STEPS_PER_FRAME):
            last_loss = train_step()
            self._step += 1
        self._loss_hist.append(max(last_loss.item(), 1e-6))

        pat_mean, pos_loss, stripe = probe_views()

        # The only two lines that touch the GPU->display path. Everything
        # spatial stays on the device; nothing here reads back.
        for h in range(N_HEADS):
            head_tex[h].update(pat_mean[h])
            self._stripe_hist[h].append(float(stripe[h]))
        strip_tex.update(pos_loss)

        update_line(loss_geom, [math.log10(v) for v in self._loss_hist], LOG_LO, LOG_HI)
        for h in range(N_HEADS):
            update_line(stripe_geoms[h], self._stripe_hist[h], 0.0, 1.0)

        if self._step % 25 < STEPS_PER_FRAME:
            print(
                f"step {self._step:4d}  loss {self._loss_hist[-1]:.4f}  "
                f"stripe {np.round(stripe, 2)}"
            )


class EdgeGUI(ImguiWindow):
    def __init__(self):
        super().__init__()
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
        imgui.text("lr:")
        imgui.same_line()
        changed, lr = imgui.slider_float("##LR", self._learning_rate, 1e-6, 1e-1)
        if changed:
            self._learning_rate = lr
            opt.lr = self._learning_rate

        self._make_title("Head Ablation")
        for i in range(N_HEADS):
            changed, val = imgui.checkbox(f"h{i}", ABLATE[i])
            if changed:
                ABLATE[i] = val
            if i != N_HEADS - 1:
                imgui.same_line()


window_flags = (
    imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.no_move
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_scrollbar
    | imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_scroll_with_mouse
)

figure.add_imgui_window(
    MenuGUI(), size=40, location="top", title=None, window_flags=window_flags
)
figure.add_imgui_window(
    EdgeGUI(),
    size=200,
    location="right",
    title="Training Params",
    window_flags=imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize,
)

figure[0, 0].camera.show_object(
    figure[0, 0].scene, view_dir=(0, 0, -1), up=(0, 1, 0), scale=0.7
)
figure[0, 0].controller.enabled = False

figure.show()

if __name__ == "__main__":
    fpl.loop.run()
