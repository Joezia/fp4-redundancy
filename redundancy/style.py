"""Figure style (spec 007 section 8): sizes, fixed colours, fonts and label language."""
import json
from pathlib import Path

PANEL_W, PANEL_H = 4.2, 3.2
DPI = 200
REAL = "#0072B2"
BASE = "#999999"
LINE = "#D55E00"
SERIES = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
BAND_ALPHA = 0.18
SHADE = ["#F2E6D0", "#D9ECF2", "#E5F2D9", "#F2D9E8"]
CJK_FONTS = ("Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP", "Source Han Sans SC",
             "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei", "PingFang SC", "Droid Sans Fallback")

FILES = {"R1": "R1_distribution", "R2": "R2_groups", "R3": "R3_mi_distance", "R4": "R4_autocorr",
         "R5": "R5_spectrum", "R6": "R6_repeats", "R7a_unit": "R7a_neighbors_unit",
         "R7a_row": "R7a_neighbors_row", "R7a_partner": "R7a_neighbors_partner", "R7b": "R7b_clusters",
         "R8": "R8_dictionary"}


def file_name(fig):
    if fig.startswith("R9"):
        return fig  # R9_<hypothesis name>
    return FILES[fig]


def setup(lang=None):
    """Pick the label language: Chinese when a CJK font is installed (or forced), else English.
    Returns (labels, language, font name or None, message for the log)."""
    import matplotlib
    from matplotlib import font_manager
    with open(Path(__file__).with_name("labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    have = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in CJK_FONTS if n in have), None)
    if lang is None:
        lang = "zh" if font else "en"
    if lang == "zh" and not font:
        lang = "en"
    if lang == "zh":
        matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        msg = f"labels: Chinese (font {font})"
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        msg = "labels: English (no CJK font found)" if not font else "labels: English (requested)"
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    return labels[lang], lang, font, msg
