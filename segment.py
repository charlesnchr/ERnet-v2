"""Segment endoplasmic reticulum images with the pre-trained ERnet model and
measure the topology of the tubular network.

    python segment.py cell.png timelapse.tif --out results

Inputs are greyscale images (PNG, JPG, TIFF) or TIFF stacks of frames. For every
input the script writes the class map, a colour rendering, the skeleton graph
drawn over the image and a summary figure. One row per frame goes to
ernet_metrics.csv. The weights are downloaded on first use.
"""

import argparse
import csv
import hashlib
import os
import sys
import urllib.request

import numpy as np
import tifffile
import torch
from PIL import Image
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "Training", "archs"), os.path.join(HERE, "Inference")]

import networkx as nx  # noqa: E402
import sknw  # noqa: E402
from graph_processing import binariseImage, build_graph, simple_analysis  # noqa: E402
from swinir_rcab_arch import SwinIR_RCAB  # noqa: E402

RELEASE = "https://github.com/charlesnchr/ERnet-v2/releases/download/v2.0/"
WEIGHTS_NAME = "20220306_ER_4class_swinir_nch1.pth"
SHA256 = {
    WEIGHTS_NAME: "9d20a526dd6f001a13fbb984d549e7547cab02cf5a7b2256b29d46f5d3a95387",
}

# Grey values of the class map, the same as in the Colab notebook and the desktop app.
BACKGROUND, SHEET, SBT, TUBULE = 0, 85, 170, 255
# The network orders its outputs as background, sheet-based tubule, tubule, sheet.
OUTPUT_TO_GREY = np.array([BACKGROUND, SBT, TUBULE, SHEET], dtype="uint8")
COLOURS = {TUBULE: (0, 200, 255), SHEET: (255, 200, 0), SBT: (255, 60, 170)}
DEGREE_COLOURS = {1: (60, 220, 90), 2: (150, 150, 150), 3: (255, 70, 60)}
MULTIWAY_COLOUR = (190, 110, 255)
WINDOW = 4  # the attention window, image sides are padded to a multiple of it

COLUMNS = ["file", "frame", "tubule_fraction", "sheet_fraction", "sbt_fraction",
           "nodes", "edges", "assortativity", "clustering", "fragments",
           "node_assembly_ratio", "edge_assembly_ratio"] + ["degree_%d" % d for d in range(1, 7)]


def download(name, dest):
    """Fetch a file from the v2.0 release and check its hash."""
    print("Downloading", RELEASE + name)
    try:
        urllib.request.urlretrieve(RELEASE + name, dest + ".part")
    except OSError as err:
        raise SystemExit("Download failed (%s). Fetch %s by hand and save it as %s." % (err, RELEASE + name, dest))
    with open(dest + ".part", "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != SHA256[name]:
        os.remove(dest + ".part")
        raise SystemExit("Checksum mismatch for %s, the download was discarded." % name)
    os.replace(dest + ".part", dest)


def load_model(weights, device):
    """Build the single-frame Swin model of the v2.0 release and load its weights."""
    if not os.path.isfile(weights):
        if os.path.basename(weights) != WEIGHTS_NAME:
            raise SystemExit("No such file: %s" % weights)
        download(WEIGHTS_NAME, weights)
    net = SwinIR_RCAB(argparse.Namespace(task="segment"), img_size=128, in_chans=1, upscale=1)
    checkpoint = torch.load(weights, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    net.load_state_dict({k.replace("module.", "", 1): v for k, v in state_dict.items()})
    return net.to(device).eval()


def read_frames(path):
    """Return the image as a float array of shape (frames, height, width)."""
    if path.lower().endswith((".tif", ".tiff")):
        img = tifffile.imread(path)
    else:
        img = Image.open(path)
        img = np.asarray(img.convert("RGB") if img.mode == "P" else img)
    img = img.astype("float64")
    if img.ndim >= 3 and img.shape[-1] <= 4:  # colour image, keep the brightest channel
        if img.shape[-1] in (2, 4):
            img = img[..., :-1]  # drop the alpha channel
        img = img.max(-1)
    if img.ndim == 2:
        img = img[None]
    if img.ndim != 3:
        raise ValueError("expected a 2D image or a stack of 2D frames, got shape %s" % (img.shape,))
    return img


def normalise(frame):
    """Stretch the 1st to 99th percentile to [0, 1] in 256 levels, as the model was trained."""
    lo, hi = np.percentile(frame, (1, 99))
    frame = np.clip((frame - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(frame)
    return (frame * 255).astype("uint8")


def classify(net, tile, device):
    """Run the network on one uint8 tile and return the class map in grey values."""
    h, w = tile.shape
    x = torch.from_numpy(tile.astype("float32") / 255)[None, None].to(device)
    pad_h, pad_w = -h % WINDOW, -w % WINDOW
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.no_grad():
        index = net(x)[0, :, :h, :w].argmax(0).cpu().numpy()
    return OUTPUT_TO_GREY[index]


def segment(net, frame, device, tile=1024, overlap=64):
    """Segment one normalised frame. Frames larger than a tile are processed in overlapping tiles."""
    h, w = frame.shape
    if h <= tile and w <= tile:
        return classify(net, frame, device)
    out = np.zeros((h, w), "uint8")
    step = tile - 2 * overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            y0, x0 = max(y - overlap, 0), max(x - overlap, 0)
            y1, x1 = min(y + step + overlap, h), min(x + step + overlap, w)
            classes = classify(net, frame[y0:y1, x0:x1], device)
            ye, xe = min(y + step, h), min(x + step, w)
            out[y:ye, x:xe] = classes[y - y0:ye - y0, x - x0:xe - x0]
    return out


def tubule_graph(classes):
    """Skeletonise the tubule class and return (skeleton graph, simplified graph)."""
    skeleton = skeletonize(binariseImage(classes) / 255).astype(np.uint16)
    graph = sknw.build_sknw(skeleton)
    return graph, build_graph(graph.nodes(), np.array(graph.edges()))


def measure(classes, graph):
    """Class fractions of the ER area and the connectivity metrics of the tubule graph."""
    counts = [int(np.sum(classes == c)) for c in (TUBULE, SHEET, SBT)]
    total = sum(counts)
    row = dict.fromkeys(COLUMNS[2:12], float("nan"))  # stays undefined where there is no ER or no tubule
    if total:
        row.update(zip(COLUMNS[2:5], [c / total for c in counts]))
    row.update(nodes=graph.number_of_nodes(), edges=0, fragments=nx.number_connected_components(graph))
    if graph.number_of_edges():
        row.update(zip(COLUMNS[5:12], simple_analysis(graph)))
    degrees = [d for _, d in graph.degree()]
    row.update(("degree_%d" % d, degrees.count(d)) for d in range(1, 7))
    return row


def colourise(classes):
    rgb = np.zeros(classes.shape + (3,), "uint8")
    for grey, colour in COLOURS.items():
        rgb[classes == grey] = colour
    return rgb


def stamp(rgb, ys, xs, colour, radius):
    """Paint the pixels (ys, xs) and their neighbours within radius."""
    h, w = rgb.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius + 1:
                rgb[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] = colour


def draw_graph(frame, skeleton_graph, graph):
    """Draw the skeleton in white and the nodes, coloured by degree, over the dimmed input."""
    rgb = np.repeat((frame * 0.45).astype("uint8")[..., None], 3, axis=2)
    for s, e in skeleton_graph.edges():
        pts = skeleton_graph[s][e]["pts"]
        rgb[pts[:, 0], pts[:, 1]] = (255, 255, 255)
    by_degree = {}
    for node, degree in graph.degree():
        if degree:
            by_degree.setdefault(degree, []).append(skeleton_graph.nodes[node]["o"])
    for degree, centres in sorted(by_degree.items()):
        centres = np.rint(np.array(centres)).astype(int)
        stamp(rgb, centres[:, 0], centres[:, 1], DEGREE_COLOURS.get(degree, MULTIWAY_COLOUR), 2)
    return rgb


def summary_figure(path, name, frame, colour, graph_rgb, rows):
    """Input, segmentation and graph of the first frame, with the metrics underneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    as_rgb = lambda c: tuple(v / 255 for v in c)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4), height_ratios=(3, 1.6), constrained_layout=True)
    titles = ("Input", "ERnet segmentation", "Tubule skeleton and junctions")
    for ax, img, title in zip(axes[0], (frame, colour, graph_rgb), titles):
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(title if len(rows) == 1 else title + " (frame 0)")
        ax.axis("off")
    axes[0, 1].legend(handles=[Patch(color=as_rgb(COLOURS[c]), label=l) for c, l in
                               ((TUBULE, "tubule"), (SHEET, "sheet"), (SBT, "sheet-based tubule"))],
                      loc="lower left", fontsize=9, framealpha=0.9)
    axes[0, 2].legend(handles=[Patch(color=as_rgb(DEGREE_COLOURS[1]), label="tip (degree 1)"),
                               Patch(color=as_rgb(DEGREE_COLOURS[3]), label="three-way junction"),
                               Patch(color=as_rgb(MULTIWAY_COLOUR), label="multi-way junction")],
                      loc="lower left", fontsize=9, framealpha=0.9)

    first = rows[0]
    t = [r["frame"] for r in rows]
    fraction_ax, graph_ax, degree_ax = axes[1]
    fractions = (("tubule_fraction", TUBULE, "tubule"), ("sheet_fraction", SHEET, "sheet"),
                 ("sbt_fraction", SBT, "sheet-based tubule"))
    if len(rows) == 1:
        fraction_ax.bar([l for _, _, l in fractions], [100 * first[k] for k, _, _ in fractions],
                        color=[as_rgb(COLOURS[c]) for _, c, _ in fractions])
        graph_ax.axis("off")
        lines = ["%s: %s" % (k.replace("_", " "), ("%.3f" % first[k]) if isinstance(first[k], float) else first[k])
                 for k in COLUMNS[5:12]]
        graph_ax.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=12, family="monospace")
    else:
        for key, c, label in fractions:
            fraction_ax.plot(t, [100 * r[key] for r in rows], color=as_rgb(COLOURS[c]), lw=2, label=label)
        fraction_ax.set_ylim(0, 100)
        graph_ax.plot(t, [r["nodes"] for r in rows], lw=2, label="nodes")
        graph_ax.plot(t, [r["edges"] for r in rows], lw=2, label="edges")
        graph_ax.set_title("Tubule graph")
        for ax in (fraction_ax, graph_ax):
            ax.set_xlabel("Frame")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(fontsize=9)
    fraction_ax.set_ylabel("% of ER area")
    fraction_ax.set_title("ER composition")
    means = [np.mean([r["degree_%d" % d] for r in rows]) for d in range(1, 7)]
    degree_ax.bar(range(1, 7), means, color=[as_rgb(DEGREE_COLOURS.get(d, MULTIWAY_COLOUR)) for d in range(1, 7)])
    degree_ax.set_xlabel("Node degree")
    degree_ax.set_ylabel("Nodes" if len(rows) == 1 else "Nodes (mean over frames)")
    degree_ax.set_title("Degree distribution")
    fig.suptitle(name)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def process(net, path, frames, out_dir, device, tile):
    """Segment and analyse the frames of one file. Returns one metrics row per frame."""
    name = os.path.splitext(os.path.basename(path))[0]
    frames = [normalise(f) for f in frames]
    classes, colours, graphs, rows = [], [], [], []
    for i, frame in enumerate(frames):
        seg = segment(net, frame, device, tile)
        skeleton_graph, graph = tubule_graph(seg)
        classes.append(seg)
        colours.append(colourise(seg))
        graphs.append(draw_graph(frame, skeleton_graph, graph))
        rows.append(dict(file=os.path.basename(path), frame=i, **measure(seg, graph)))
        print("  %s frame %d/%d: %d nodes, %d edges" % (name, i + 1, len(frames), rows[-1]["nodes"], rows[-1]["edges"]))

    base = os.path.join(out_dir, name)
    if len(frames) == 1:
        for suffix, img in (("classes", classes[0]), ("colour", colours[0]), ("graph", graphs[0])):
            Image.fromarray(img).save("%s_%s.png" % (base, suffix))
    else:
        for suffix, stack in (("classes", classes), ("colour", colours), ("graph", graphs)):
            tifffile.imwrite("%s_%s.tif" % (base, suffix), np.stack(stack), compression="zlib")
    summary_figure(base + "_summary.png", os.path.basename(path), frames[0], colours[0], graphs[0], rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", help="images or TIFF stacks to segment")
    p.add_argument("--out", default="results", help="output folder")
    p.add_argument("--weights", default=os.path.join(HERE, WEIGHTS_NAME), help="path to the model file")
    p.add_argument("--tile", type=int, default=1024, help="largest side processed in one pass, in pixels")
    p.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is available")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    net = load_model(args.weights, device)
    os.makedirs(args.out, exist_ok=True)
    print("Segmenting on", device)

    rows = []
    for path in args.images:
        try:
            frames = read_frames(path)
        except Exception as err:  # the readers raise many different types for a bad file
            print("Skipping %s: %s" % (path, err))
            continue
        rows += process(net, path, frames, args.out, device, args.tile)
    if not rows:
        raise SystemExit("Nothing was segmented.")
    with open(os.path.join(args.out, "ernet_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v if not isinstance(v, float) else "" if np.isnan(v) else "%.5f" % v
                             for k, v in row.items()})
    print("Wrote the results to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
