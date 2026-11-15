"""Regenerate the README figures from the test acquisitions of the v2.0 release.

    python fig/make_figures.py

Every image is the output of segment.py with the released weights. The script
downloads the four test acquisitions (38 MB) and the ten synthetic networks of
the paper's ground-truth test (1 MB, from figshare), segments them if the
results folder does not have them yet, and draws the figures. Segmenting the
37 frames takes about 25 minutes on a 4-core CPU.
Needs ffmpeg for the GIF, and the Lato font for the same look.
"""

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import segment  # noqa: E402

BG = (13, 17, 23)
INK = (240, 246, 252)
MUTED = (139, 148, 158)
TEST_IMAGES = os.path.join(ROOT, "test_images")
ACQUISITIONS = ("TestImage1.png", "TestImage2.png", "TestImage3-stack.tif", "TestImage4-stack.tif")
TIMELAPSE = "TestImage4-stack.tif"
SYNTHETIC_URL = "https://ndownloader.figshare.com/files/38999795"  # ERnet datasets, doi:10.6084/m9.figshare.21975878

# (x, y) of the 300 px crops of the time-lapse shown in the animation
HERO_CROPS = ((10, 580), (330, 290), (320, 600))
# file, frame and (x, y) of the 380 px crops in the results grid
GRID_CROPS = (("TestImage1.png", 0, (560, 520)), ("TestImage2.png", 0, (300, 300)),
              ("TestImage3-stack.tif", 0, (560, 560)), ("TestImage4-stack.tif", 0, (60, 520)))
LEGEND = (("Tubules", segment.COLOURS[segment.TUBULE]), ("Sheets", segment.COLOURS[segment.SHEET]),
          ("Sheet-based tubules", segment.COLOURS[segment.SBT]))
NODE_LEGEND = (("Tubule tip", segment.DEGREE_COLOURS[1]), ("Three-way junction", segment.DEGREE_COLOURS[3]),
               ("Multi-way junction", segment.MULTIWAY_COLOUR))


def font(size, weight="Regular"):
    for path in ("/usr/share/fonts/truetype/lato/Lato-%s.ttf" % weight,
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if weight != "Regular" else "")):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ----------------------------------------------------------------------------- data

def ensure_results(results):
    """Download the test acquisitions and segment them, unless the results are already there."""
    os.makedirs(TEST_IMAGES, exist_ok=True)
    paths = [os.path.join(TEST_IMAGES, name) for name in ACQUISITIONS]
    for name, path in zip(ACQUISITIONS, paths):
        if not os.path.isfile(path):
            segment.download(name, path)
    done = [os.path.isfile(os.path.join(results, "%s_classes.%s" % (os.path.splitext(n)[0], n[-3:])))
            for n in ACQUISITIONS]
    if not all(done):
        subprocess.run([sys.executable, os.path.join(ROOT, "segment.py"), *paths, "--out", results], check=True)


def load(results, name):
    """Return (input, classes, graph) of an acquisition as arrays with a leading frame axis."""
    stem = os.path.splitext(name)[0]
    frames = np.stack([segment.normalise(f) for f in segment.read_frames(os.path.join(TEST_IMAGES, name))])
    out = [frames]
    for kind in ("classes", "graph"):
        path = os.path.join(results, "%s_%s" % (stem, kind))
        out.append(tifffile.imread(path + ".tif") if len(frames) > 1 else np.asarray(Image.open(path + ".png"))[None])
    return out


def metrics(results, name):
    with open(os.path.join(results, "ernet_metrics.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["file"] == name]
    return {k: np.array([float(r[k]) for r in rows]) for k in segment.COLUMNS[2:]}


def overlay(frame, classes, texture=0.5):
    """The class colours, shaded by the input intensity so that the texture stays visible."""
    shade = 1 - texture + texture * (frame.astype("float32") / 255)[..., None]
    rgb = segment.colourise(classes).astype("float32") * shade
    background = classes == segment.BACKGROUND
    rgb[background] = 0.3 * frame[background][:, None]
    return rgb.astype("uint8")


def synthetic_benchmark(results):
    """Segment the ten synthetic networks of the paper and count nodes and edges against the ground truth.

    Returns (rows, example) where rows holds (nodes, edges, true nodes, true edges) per network
    and example is (input, graph drawing) of the first network.
    """
    import torch

    folder = os.path.join(TEST_IMAGES, "synthetic")
    if not os.path.isfile(os.path.join(folder, "9_GT.png")):
        os.makedirs(folder, exist_ok=True)
        print("Downloading", SYNTHETIC_URL)
        with urllib.request.urlopen(SYNTHETIC_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for member in archive.namelist():
            name = os.path.basename(member)
            if name.endswith(("_IN.png", "_GT.png")) and not name.startswith("."):
                with open(os.path.join(folder, name), "wb") as f:
                    f.write(archive.read(member))

    cache = os.path.join(results, "synthetic")
    os.makedirs(cache, exist_ok=True)
    net, rows, example = None, [], None
    for i in range(10):
        frame = segment.normalise(segment.read_frames(os.path.join(folder, "%d_IN.png" % i))[0])
        truth = np.asarray(Image.open(os.path.join(folder, "%d_GT.png" % i)).convert("L"))
        path = os.path.join(cache, "%d_classes.png" % i)
        if not os.path.isfile(path):
            if net is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                net = segment.load_model(os.path.join(ROOT, segment.WEIGHTS_NAME), device)
            Image.fromarray(segment.segment(net, frame, device)).save(path)
        classes = np.asarray(Image.open(path))
        skeleton_graph, graph = segment.tubule_graph(classes)
        _, true_graph = segment.tubule_graph(truth)
        rows.append((graph.number_of_nodes(), graph.number_of_edges(),
                     true_graph.number_of_nodes(), true_graph.number_of_edges()))
        if i == 0:
            example = (frame, truth, segment.draw_graph(frame, skeleton_graph, graph))
    return np.array(rows, dtype=float), example


# ----------------------------------------------------------------------------- drawing

def pill(draw, xy, text, fnt, anchor="la", fill=(0, 0, 0), ink=INK):
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=fnt, anchor=anchor)
    draw.rounded_rectangle((x0 - 8, y0 - 5, x1 + 8, y1 + 5), radius=6, fill=fill)
    draw.text(xy, text, font=fnt, anchor=anchor, fill=ink)


def legend(draw, x, y, entries, fnt, dot=False, gap=26):
    """Draw colour swatches with labels from x to the right and return the x after the last one."""
    for label, colour in entries:
        if dot:
            draw.ellipse((x, y + 3, x + 12, y + 15), fill=colour)
        else:
            draw.rounded_rectangle((x, y + 3, x + 12, y + 15), radius=2, fill=colour)
        draw.text((x + 19, y + 9), label, font=fnt, fill=INK, anchor="lm")
        x = draw.textbbox((x + 19, y + 9), label, font=fnt, anchor="lm")[2] + gap
    return x


def crop(img, xy, size):
    x, y = xy
    return img[y:y + size, x:x + size]


def as_rgb(img):
    return img if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)


def wipe(layers, positions):
    """Stack the layers left to right: layers[i + 1] covers the part left of positions[i]."""
    tile = Image.fromarray(as_rgb(layers[0]))
    for layer, split in zip(layers[1:], positions):
        if split > 0:
            tile.paste(Image.fromarray(as_rgb(layer)).crop((0, 0, split, tile.height)), (0, 0))
    return tile


def make_hero(results, out, size=300, fps=12, hold_frames=3):
    """The time-lapse, wiping from the input to the segmentation and on to the skeleton graph.

    Every frame of the time-lapse is shown for hold_frames frames of the animation.
    """
    frames, classes, graphs = load(results, TIMELAPSE)
    n = len(frames)
    order = list(range(n)) + list(range(n - 2, 0, -1))  # play forwards, then backwards
    ease = lambda k: [(1 - np.cos(np.pi * t)) / 2 for t in np.linspace(0, 1, k)]
    hold = lambda value, seconds: [value] * int(fps * seconds)
    sweep = ease(int(fps * 1.7))
    # p1 reveals the segmentation, p2 the graph
    p1 = hold(0, 0.8) + sweep + hold(1, 1.3) + hold(1, 1.7) + hold(1, 1.5) + sweep[::-1]
    p2 = hold(0, 0.8) + hold(0, 1.7) + hold(0, 1.3) + sweep + hold(1, 1.5) + sweep[::-1]
    labels = ("SIM TIME-LAPSE", "SEGMENTATION", "SKELETON GRAPH")

    gap, margin, caption_h = 12, 16, 44
    width = 2 * margin + len(HERO_CROPS) * size + (len(HERO_CROPS) - 1) * gap
    label, small = font(15, "Bold"), font(15)
    tmp = tempfile.mkdtemp()
    try:
        for i, (a, b) in enumerate(zip(p1, p2)):
            f = order[(i // hold_frames) % len(order)]
            canvas = Image.new("RGB", (width, margin + size + caption_h), BG)
            draw = ImageDraw.Draw(canvas)
            for j, xy in enumerate(HERO_CROPS):
                raw = crop(frames[f], xy, size)
                layers = (raw, overlay(raw, crop(classes[f], xy, size), 0.3), crop(graphs[f], xy, size))
                s1, s2 = int(round(a * size)), int(round(b * size))
                tile = wipe(layers, (s1, s2))
                td = ImageDraw.Draw(tile)
                for split in (s1, s2):
                    if 0 < split < size:
                        td.line((split, 0, split, size), fill=INK, width=2)
                if j == 0:  # name the layer at each edge of the first tile, where there is room for it
                    left = labels[2] if s2 > 160 else labels[1] if s1 > 150 and s2 == 0 else None
                    right = labels[0] if size - s1 > 160 else labels[1] if s1 == size and 0 < s2 < size - 150 else None
                    if left:
                        pill(td, (12, 12), left, label)
                    if right:
                        pill(td, (size - 12, 12), right, label, anchor="ra")
                canvas.paste(tile, (margin + j * (size + gap), margin))
            y = margin + size + 13
            x = legend(draw, margin, y, LEGEND, small)
            if b > 0.5:
                legend(draw, x + 10, y, NODE_LEGEND, small, dot=True)
            canvas.save(os.path.join(tmp, "f%04d.png" % i))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
             "-vf", "split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
             out], check=True)
    finally:
        shutil.rmtree(tmp)


def make_grid(results, out, size=380):
    """Input, segmentation and skeleton graph for one crop of each test acquisition."""
    gap, margin, left, head, foot = 10, 20, 44, 50, 46
    rows = ("INPUT", "SEGMENTATION", "SKELETON GRAPH")
    width = left + margin + len(GRID_CROPS) * size + (len(GRID_CROPS) - 1) * gap + margin
    height = head + len(rows) * size + (len(rows) - 1) * gap + foot
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    for i, (name, frame, xy) in enumerate(GRID_CROPS):
        frames, classes, graphs = load(results, name)
        x0 = left + margin + i * (size + gap)
        draw.text((x0, 12), os.path.splitext(name)[0], font=font(24, "Bold"), fill=INK)
        raw = crop(frames[frame], xy, size)
        for row, img in enumerate((raw, overlay(raw, crop(classes[frame], xy, size)), crop(graphs[frame], xy, size))):
            canvas.paste(Image.fromarray(as_rgb(img)), (x0, head + row * (size + gap)))
    for row, text in enumerate(rows):
        strip = Image.new("RGB", (size, 36), BG)
        ImageDraw.Draw(strip).text((size // 2, 18), text, font=font(22, "Bold"), fill=INK if row else MUTED, anchor="mm")
        canvas.paste(strip.rotate(90, expand=True), (margin - 2, head + row * (size + gap)))
    y = height - foot + 14
    x = legend(draw, left + margin, y, LEGEND, font(19), gap=34)
    legend(draw, x + 30, y, NODE_LEGEND, font(19), dot=True, gap=34)
    canvas.save(out, optimize=True)


def make_social(results, out):
    """1280x640 image for the repository's social preview."""
    w, h, size = 1280, 640, 300
    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 52), "ERnet", font=font(84, "Black"), fill=INK)
    draw.text((66, 156), "Segment the endoplasmic reticulum and measure its network topology",
              font=font(31, "Medium"), fill=MUTED)
    draw.text((w - 64, 100), "Nature Methods 2023", font=font(22), fill=MUTED, anchor="ra")
    frames, classes, graphs = load(results, TIMELAPSE)
    xy = HERO_CROPS[1]
    raw = crop(frames[0], xy, size)
    tiles = ((raw, "SIM image"), (overlay(raw, crop(classes[0], xy, size)), "Tubules and sheets"),
             (crop(graphs[0], xy, size), "Network graph"))
    gap = 26
    x = (w - (len(tiles) * size + (len(tiles) - 1) * gap)) // 2
    for img, caption in tiles:
        canvas.paste(Image.fromarray(as_rgb(img)), (x, 236))
        draw.text((x, 236 + size + 14), caption, font=font(22, "Bold"), fill=INK)
        x += size + gap
    canvas.save(out, optimize=True)


def style_axes(ax, muted):
    ax.tick_params(colors=muted, labelsize=12, length=0)
    ax.grid(axis="y", color=muted, alpha=0.18, lw=0.8)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(muted)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for weight in ("Regular", "Bold"):
        path = "/usr/share/fonts/truetype/lato/Lato-%s.ttf" % weight
        if os.path.isfile(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = "Lato"
    return plt


def make_dynamics(results, out):
    """What the 30-frame time-lapse gives: composition, graph size and junction types per frame."""
    plt = pyplot()
    m = metrics(results, TIMELAPSE)
    t = np.arange(len(m["nodes"]))
    bg, ink, muted = [tuple(v / 255 for v in c) for c in (BG, INK, MUTED)]
    colour = lambda c: tuple(v / 255 for v in c)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=150, facecolor=bg)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.17, wspace=0.26)
    ax = axes[0]
    for key, (label, c) in zip(("tubule_fraction", "sheet_fraction", "sbt_fraction"), LEGEND):
        ax.plot(t, 100 * m[key], color=colour(c), lw=2.2)
        ax.text(t[-1] + 0.6, 100 * m[key][-1], label, color=ink, fontsize=12, va="center")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, t[-1] + 19)
    ax.set_title("ER composition (% of ER area)", color=ink, fontsize=14, fontweight="bold", loc="left")

    ax = axes[1]
    for key, label, c in (("nodes", "Nodes", "#3987e5"), ("edges", "Edges", "#d95926")):
        ax.plot(t, m[key], color=c, lw=2.2)
        ax.text(t[-1] + 0.6, m[key][-1], label, color=ink, fontsize=12, va="center")
    ax.set_xlim(0, t[-1] + 5)
    ax.set_title("Size of the tubule graph", color=ink, fontsize=14, fontweight="bold", loc="left")

    ax = axes[2]
    junctions = m["degree_3"] + m["degree_4"] + m["degree_5"] + m["degree_6"]
    ax.plot(t, 100 * m["degree_3"] / junctions, color=colour(segment.DEGREE_COLOURS[3]), lw=2.2)
    ax.plot(t, 100 * (junctions - m["degree_3"]) / junctions, color=colour(segment.MULTIWAY_COLOUR), lw=2.2)
    ax.text(t[-1] + 0.6, 100 * m["degree_3"][-1] / junctions[-1], "Three-way", color=ink, fontsize=12, va="center")
    ax.text(t[-1] + 0.6, 100 * (1 - m["degree_3"][-1] / junctions[-1]), "Multi-way", color=ink, fontsize=12, va="center")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, t[-1] + 8)
    ax.set_title("Junction types (% of junctions)", color=ink, fontsize=14, fontweight="bold", loc="left")

    for ax in axes:
        ax.set_facecolor(bg)
        ax.set_xlabel("Frame", color=muted, fontsize=12)
        ax.set_xticks(range(0, len(t), 5))
        style_axes(ax, muted)
    fig.savefig(out, facecolor=bg)
    plt.close(fig)


def make_benchmark(results, out):
    """Nodes and edges found in the ten synthetic networks against their ground truth."""
    plt = pyplot()
    rows, (frame, truth, graph) = synthetic_benchmark(results)
    bg, ink, muted = [tuple(v / 255 for v in c) for c in (BG, INK, MUTED)]
    error = 100 * np.abs(rows[:, :2] - rows[:, 2:]) / rows[:, 2:]
    print("Synthetic networks, error in node count (%):", np.round(error[:, 0], 1), "median %.1f" % np.median(error[:, 0]))
    print("Synthetic networks, error in edge count (%):", np.round(error[:, 1], 1), "median %.1f" % np.median(error[:, 1]))

    fig = plt.figure(figsize=(13.5, 3.5), dpi=150, facecolor=bg)
    side = 0.78 * 3.5 / 13.5
    view = (slice(96, 352), slice(96, 352))  # the central 256 px, shown enlarged
    panels = ((frame[view], "Synthetic image"), (truth[view], "Ground truth"), (graph[view], "ERnet graph"))
    for i, (img, title) in enumerate(panels):
        ax = fig.add_axes((0.012 + i * (side + 0.012), 0.06, side, 0.78))
        ax.imshow(as_rgb(img), interpolation="nearest")
        ax.set_title(title, color=ink, fontsize=14, fontweight="bold", loc="left", pad=8)
        ax.axis("off")
    left = 0.012 + 3 * (side + 0.012) + 0.05
    ax = fig.add_axes((left, 0.2, 0.985 - left, 0.64), facecolor=bg)
    x = np.arange(1, len(rows) + 1)
    ax.axhline(100, color=muted, lw=1.2, dashes=(4, 3))
    ax.text(x[-1] + 0.45, 99.2, "ground truth", color=muted, fontsize=11, va="top", ha="right")
    for column, name, c, shift in ((0, "Nodes", "#3987e5", -0.13), (1, "Edges", "#f0764a", 0.13)):
        ax.plot(x + shift, 100 * rows[:, column] / rows[:, column + 2], "o", color=c, ms=7, label=name)
    ax.set_xticks(x)
    ax.set_xlim(0.4, x[-1] + 0.6)
    ax.set_ylim(90, 125)
    ax.set_xlabel("Synthetic network", color=muted, fontsize=12)
    ax.set_title("Count found by ERnet (% of ground truth)", color=ink, fontsize=14, fontweight="bold", loc="left", pad=8)
    ax.legend(ncol=2, fontsize=12, frameon=False, labelcolor=ink, loc="upper left", handletextpad=0.2)
    style_axes(ax, muted)
    fig.savefig(out, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(ROOT, "results"), help="folder written by segment.py")
    p.add_argument("--out", default=HERE)
    args = p.parse_args()
    ensure_results(args.results)
    make_hero(args.results, os.path.join(args.out, "hero.gif"))
    make_grid(args.results, os.path.join(args.out, "results.png"))
    make_dynamics(args.results, os.path.join(args.out, "dynamics.png"))
    make_benchmark(args.results, os.path.join(args.out, "benchmark.png"))
    make_social(args.results, os.path.join(args.out, "social-preview.png"))
