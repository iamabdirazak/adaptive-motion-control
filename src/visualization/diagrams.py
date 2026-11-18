"""Documentation diagrams: the dynamics model as a file, and the controller architecture.

The layout: a soft frame with a title, a symmetric row of raised
cards, each with an icon, a name and a short description, and small arrows
between them.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from PIL import Image

from src.paths import ROOT
from src.visualization import style

DOCS = ROOT / "docs"
LIGHT, SOFT, MEDIUM, RICH = (style.SHADES[k] for k in ("light", "soft", "medium", "rich"))
FRAME, CARD, ARROW = style.NEUTRAL["background"], style.NEUTRAL["surface"], style.NEUTRAL["tertiary"]

CARD_W, CARD_H, GAP = 2.6, 3.4, 0.41  # portrait cards, like the UAV figure
MARGIN = 0.4  # around the row of cards, equal to CARD_Y
CARD_Y = 0.4
CARD_RADIUS, FRAME_RADIUS = 0.27, 0.38  # in proportion to the UAV figure's cards and frame
HEIGHT = CARD_Y + CARD_H + 1.0  # room above the cards for the title
ICON_Y = CARD_Y + 2.25
NAME_Y, SUBTITLE_Y = CARD_Y + 1.05, CARD_Y + 0.65
MIDDLE = CARD_Y + CARD_H / 2


# Drawing primitives ---------------------------------------------------------

def rounded(axis, x, y, w, h, color, radius=0.12, z=3):
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                  facecolor=color, edgecolor="none", zorder=z))


def wave(axis, cx, cy, width, amplitude, color, cycles=1.0, lw=2.2, z=4):
    """A smooth move: a minimum-jerk rise, as a joint angle traces it."""
    x = np.linspace(0, 1, 120)
    y = 10 * x**3 - 15 * x**4 + 6 * x**5
    axis.plot(cx - width / 2 + x * width, cy - amplitude + 2 * amplitude * y, color=color, linewidth=lw,
              solid_capstyle="round", zorder=z)


def dot(axis, x, y, r, color, alpha=1.0, z=5):
    axis.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", alpha=alpha, zorder=z))


# Icons, each centred on (cx, cy) ------------------------------------------

INK = style.NEUTRAL["ink"]


def outline(axis, x, y, w, h, color, radius=0.1, lw=1.6, z=3, dash=None):
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor="none",
                                  edgecolor=color, linewidth=lw, linestyle=dash or "-", zorder=z))


def link(axis, a, b, color, width=9, z=4, alpha=1.0):
    axis.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=width, solid_capstyle="round", alpha=alpha, zorder=z)


def icon_input(axis, cx, cy):
    """Encoder readings arriving one frame after another, each holding a joint's smooth move."""
    for offset, alpha in ((-0.28, 0.3), (-0.14, 0.55), (0.0, 1.0)):
        rounded(axis, cx - 0.5 + offset, cy - 0.34 - offset, 0.95, 0.68, RICH, 0.12, z=3 + alpha)
        axis.patches[-1].set_alpha(alpha)
    wave(axis, cx - 0.02, cy, 0.6, 0.18, LIGHT, lw=2.4, z=6)


def arm(axis, cx, cy, color, joint_color, s=1.0, z=4, alpha=1.0):
    """A cobot in side view: base, two links and a wrist, joints marked."""
    base, shoulder = (cx - 0.42 * s, cy - 0.42 * s), (cx - 0.42 * s, cy - 0.1 * s)
    elbow, wrist = (cx + 0.02 * s, cy + 0.3 * s), (cx + 0.42 * s, cy + 0.02 * s)
    rounded(axis, base[0] - 0.2 * s, base[1] - 0.1 * s, 0.4 * s, 0.14 * s, color, 0.05 * s, z=z)
    axis.patches[-1].set_alpha(alpha)
    for a, b in ((base, shoulder), (shoulder, elbow), (elbow, wrist)):
        link(axis, a, b, color, 8 * s, z, alpha)
    for joint in (shoulder, elbow, wrist):
        dot(axis, *joint, 0.075 * s, joint_color, alpha, z + 1)
    return wrist


def icon_nominal(axis, cx, cy):
    """The arm as its datasheet describes it: links, joints and masses, no payload."""
    arm(axis, cx, cy, MEDIUM, LIGHT)


def icon_residual(axis, cx, cy):
    """A small network turning the arm's state into features."""
    layers = [np.linspace(-0.36, 0.36, 4), np.linspace(-0.45, 0.45, 5), np.linspace(-0.36, 0.36, 4)]
    xs = (cx - 0.5, cx, cx + 0.5)
    for (x0, ys0), (x1, ys1) in zip(zip(xs, layers), zip(xs[1:], layers[1:])):
        for y0 in ys0:
            for y1 in ys1:
                axis.plot([x0, x1], [cy + y0, cy + y1], color=RICH, linewidth=0.8, alpha=0.45, zorder=3)
    for x, ys, color in zip(xs, layers, (MEDIUM, RICH, LIGHT)):
        for y in ys:
            dot(axis, x, cy + y, 0.075, color)


def icon_adaptation(axis, cx, cy):
    """Least squares at every step: recent measurements count, older ones fade, and the fit follows."""
    rng = np.random.default_rng(3)
    xs = np.linspace(-0.55, 0.5, 11)
    for i, x in enumerate(xs):
        y = 0.55 * x + 0.06 * rng.standard_normal()
        dot(axis, cx + x, cy + y, 0.055, SOFT, alpha=0.2 + 0.8 * i / (len(xs) - 1))
    axis.plot([cx - 0.62, cx + 0.6], [cy - 0.45, cy + 0.2], color=RICH, linewidth=1.6, alpha=0.35,
              linestyle=(0, (2, 2)), zorder=4)                      # the estimate before
    axis.plot([cx - 0.62, cx + 0.6], [cy - 0.34, cy + 0.33], color=LIGHT, linewidth=2.4,
              solid_capstyle="round", zorder=6)                     # the estimate now


def icon_control(axis, cx, cy):
    """A joint driven along its reference path, with the torque that keeps it there."""
    t = np.linspace(0, 1, 60)
    axis.plot(cx - 0.6 + 1.2 * t, cy - 0.38 + 0.36 * (10 * t**3 - 15 * t**4 + 6 * t**5), color=MEDIUM,
              linewidth=1.6, linestyle=(0, (2, 2)), zorder=3)       # the reference
    dot(axis, cx, cy + 0.04, 0.2, RICH, z=5)
    dot(axis, cx, cy + 0.04, 0.07, LIGHT, z=6)
    a = np.radians(np.linspace(200, 470, 60))
    axis.plot(cx + 0.36 * np.cos(a), cy + 0.04 + 0.36 * np.sin(a), color=LIGHT, linewidth=2.2,
              solid_capstyle="round", zorder=6)
    tip = (cx + 0.36 * np.cos(a[-1]), cy + 0.04 + 0.36 * np.sin(a[-1]))
    heading = a[-1] + np.pi / 2
    axis.add_patch(Polygon([(tip[0] + 0.13 * np.cos(heading), tip[1] + 0.13 * np.sin(heading)),
                            (tip[0] + 0.09 * np.cos(heading + 2.2), tip[1] + 0.09 * np.sin(heading + 2.2)),
                            (tip[0] + 0.09 * np.cos(heading - 2.2), tip[1] + 0.09 * np.sin(heading - 2.2))],
                           facecolor=LIGHT, edgecolor="none", zorder=7))


def icon_output(axis, cx, cy):
    """Six joint torques, one per motor, sent a thousand times a second."""
    heights = (0.62, 0.86, 0.5, 0.3, 0.42, 0.22)
    for i, h in enumerate(heights):
        x = cx - 0.55 + i * 0.22
        rounded(axis, x, cy - 0.45, 0.14, h, (RICH, RICH, MEDIUM, MEDIUM, SOFT, LIGHT)[i], 0.05, z=4)
    axis.plot([cx - 0.62, cx + 0.62], [cy - 0.47, cy - 0.47], color=style.NEUTRAL["tertiary"], linewidth=1, zorder=3)


def diagram(title: str, cards: list[tuple], name: str) -> None:
    style.apply()
    width = len(cards) * CARD_W + (len(cards) - 1) * GAP + 2 * MARGIN
    figure, axis = plt.subplots(figsize=(width / 1.45, HEIGHT / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, HEIGHT)
    axis.set_aspect("equal")
    axis.axis("off")

    rounded(axis, 0, 0, width, HEIGHT, FRAME, FRAME_RADIUS, z=1)
    axis.text(width / 2, HEIGHT - 0.5, title, ha="center", va="center", fontsize=13,
              fontweight="medium", color=style.NEUTRAL["ink"], zorder=6)

    for index, (name_text, subtitle, draw) in enumerate(cards):
        x = MARGIN + index * (CARD_W + GAP)
        centre = x + CARD_W / 2
        rounded(axis, x, CARD_Y, CARD_W, CARD_H, CARD, CARD_RADIUS, z=2)
        draw(axis, centre, ICON_Y)
        axis.text(centre, NAME_Y, name_text, ha="center", va="center", fontsize=10,
                  fontweight="medium", color=style.NEUTRAL["ink"], zorder=6)
        axis.text(centre, SUBTITLE_Y, subtitle, ha="center", va="center", fontsize=7.4,
                  color=style.NEUTRAL["secondary"], zorder=6)
        if index:
            mid = x - GAP / 2
            axis.add_patch(Polygon([[mid - 0.07, MIDDLE - 0.12], [mid - 0.07, MIDDLE + 0.12], [mid + 0.09, MIDDLE]],
                                   facecolor=ARROW, edgecolor="none", zorder=3))

    style.save(figure, DOCS, name, panel=False)


def architecture() -> None:
    diagram("Model Architecture", [
        ("Input", "Encoder angles", icon_input),
        ("Nominal Model", "Datasheet dynamics", icon_nominal),
        ("Learned Residual", "Feature network", icon_residual),
        ("Adaptation", "Least squares, 1 kHz", icon_adaptation),
        ("Control", "Computed torque", icon_control),
        ("Output", "Joint torques", icon_output),
    ], "architecture")


# The model as a file -------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as Shape
from matplotlib.patches import PathPatch


def fill(axis, patch, colors, angle=90, z=3):
    """Fill a patch with a linear gradient; angle 90 runs bottom to top."""
    axis.add_patch(patch)
    patch.set_facecolor("none")
    patch.set_edgecolor("none")
    x0, y0, x1, y1 = patch.get_path().transformed(patch.get_patch_transform()).get_extents().bounds
    x1, y1 = x0 + x1, y0 + y1
    t = np.radians(angle)
    gx, gy = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    ramp = gx * np.cos(t) + gy * np.sin(t)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    image = axis.imshow(ramp, extent=(x0, x1, y0, y1), origin="lower", zorder=z,
                        cmap=LinearSegmentedColormap.from_list("", colors), interpolation="bicubic")
    image.set_clip_path(patch)


def corner(points, centre, radius, start, end):
    angles = np.linspace(np.radians(start), np.radians(end), 12)
    points += [(centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)) for a in angles]


def page(x, y, w, h, fold, radius):
    """A document outline with rounded corners and a folded top-right corner."""
    points = []
    corner(points, (x + radius, y + radius), radius, 180, 270)
    corner(points, (x + w - radius, y + radius), radius, 270, 360)
    points += [(x + w, y + h - fold), (x + w - fold, y + h)]
    corner(points, (x + radius, y + h - radius), radius, 90, 180)
    return PathPatch(Shape(points + [points[0]], closed=True))


def squircle(x, y, size, radius):
    return FancyBboxPatch((x, y), size, size, boxstyle=f"round,pad=0,rounding_size={radius}")


def onnx_mark() -> np.ndarray:
    """The ONNX mark from docs/onnx.png, without its wordmark, recoloured in the accent.

    Lines take the brightest shade. Each facet keeps its original shading, mapped
    from the deepest shade for the darkest facet to the base shade for the lightest.
    """
    source = np.asarray(Image.open(DOCS / "onnx.png").convert("RGBA")).astype(float) / 255
    mark = source[:575]                                                        # above the wordmark
    rows, columns = np.where(mark[..., 3] > 0.05)
    mark = mark[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    lightness, alpha = mark[..., :3].mean(axis=2), mark[..., 3]
    line = lightness < 0.5
    facets = LinearSegmentedColormap.from_list("", ["#7A2633", MEDIUM, RICH])  # facet shading, deepest first
    shade = np.clip((lightness - 0.6) / 0.38, 0, 1)                           # facet greys span about 0.6 to 0.98
    image = np.empty_like(mark)
    image[..., :3] = np.where(line[..., None], matplotlib.colors.to_rgb(LIGHT), facets(shade)[..., :3])
    image[..., 3] = np.where(line, alpha, alpha * 0.9)
    return image


CAPTION = "dynamics.onnx  \u00b7  7,775 parameters  \u00b7  34 KB"


def model_file() -> None:
    """The released model shown as what it is: a single ONNX file."""
    style.apply()
    width, height = 6 * CARD_W + 5 * GAP + 2 * MARGIN, 7.2
    figure, axis = plt.subplots(figsize=(width / 1.45, height / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal")
    axis.axis("off")
    rounded(axis, 0, 0, width, height, FRAME, FRAME_RADIUS, z=1)

    w, h, fold = 2.55, 3.4, 0.62                                              # document proportions, 3:4
    x, y = width / 2 - w / 2, 2.5
    cx = width / 2

    for spread, alpha in ((0.32, 0.04), (0.2, 0.06), (0.1, 0.09), (0.03, 0.14)):   # soft lift
        shadow = page(x - spread, y - spread - 0.12, w + 2 * spread, h + 2 * spread, fold + spread, 0.2 + spread)
        shadow.set(facecolor="black", edgecolor="none", alpha=alpha, zorder=2)
        axis.add_patch(shadow)

    sheet = page(x, y, w, h, fold, 0.2)
    fill(axis, sheet, ["#1D1D20", "#2A2A2E", "#38383D"], z=3)                # lit from above
    rim = page(x, y, w, h, fold, 0.2)
    rim.set(facecolor="none", edgecolor="#4A4A50", linewidth=1, zorder=4)
    axis.add_patch(rim)
    flap = PathPatch(Shape([(x + w - fold, y + h), (x + w - fold, y + h - fold + 0.1),
                            (x + w - fold + 0.1, y + h - fold), (x + w, y + h - fold),
                            (x + w - fold, y + h)], closed=True))
    fill(axis, flap, ["#3E3E44", "#5A5A61"], angle=45, z=4)

    mark = onnx_mark()
    size = 1.3                                                                 # about half the page width
    aspect = mark.shape[0] / mark.shape[1]
    label = 0.3                                                                # cap height of the label
    block = size * aspect + 0.36 + label                                       # mark, gap, label
    top = y + h * 0.47 + block / 2                                             # centred, just below the middle
    axis.imshow(mark, extent=(cx - size / 2, cx + size / 2, top - size * aspect, top),
                zorder=6, interpolation="lanczos")
    axis.text(cx, top - block + label / 2, "ONNX", ha="center", va="center", fontsize=15,
              fontweight="medium", color="#8E8E93", alpha=0.85, zorder=6)    # quieter than the mark

    axis.text(width / 2, 1.7, "Adaptive Control Model", ha="center", va="center", fontsize=14,
              fontweight="semibold", color=style.NEUTRAL["ink"], zorder=6)
    axis.text(width / 2, 1.25, CAPTION, ha="center", va="center", fontsize=9,
              color=style.NEUTRAL["secondary"], zorder=6)

    style.save(figure, DOCS, "model", panel=False)


if __name__ == "__main__":
    model_file()
    architecture()
