"""Figure style (spec 007 section 8, second version): sizes, fixed colours, font sizes, labels."""
import json
from pathlib import Path

PANEL_W, PANEL_H = 6.5, 4.5           # inches per panel
DPI = 300
REAL = "#0072B2"                       # measured
REAL2 = "#E69F00"                      # second measured curve
BASE = "#808080"                       # baseline (thin solid)
LINE = "#D55E00"                       # break-even (dashed)
GUIDE = "#808080"                      # unit boundary (dotted)
LIGHT = "#9CC7E4"                      # lower segment of stacked bars
BAND_ALPHA = 0.2
LW_REAL, LW_BASE, LW_LINE = 2.4, 1.8, 2.0
MS = 6
FONT = {"tick": 12, "label": 14, "title": 15, "legend": 12, "suptitle": 16}

FILES = {"R1": "R1_distribution", "R2": "R2_groups", "R3": "R3_mi_distance", "R4": "R4_autocorr",
         "R5": "R5_spectrum", "R6": "R6_repeats", "R7a_unit": "R7a_neighbors_unit",
         "R7a_row": "R7a_neighbors_row", "R7a_partner": "R7a_neighbors_partner", "R7b": "R7b_clusters",
         "R8": "R8_dictionary"}


def file_name(fig):
    if fig.startswith("R9"):
        return fig  # R9_<hypothesis name>
    return FILES[fig]


def setup():
    """English labels, DejaVu Sans, the minimum sizes of section 1.2. Returns (labels, log message)."""
    import matplotlib
    with open(Path(__file__).with_name("labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    rc = matplotlib.rcParams
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["DejaVu Sans"]
    rc["mathtext.fontset"] = "dejavusans"
    rc["axes.unicode_minus"] = True
    rc["pdf.fonttype"] = 42
    rc["xtick.labelsize"] = rc["ytick.labelsize"] = FONT["tick"]
    rc["axes.labelsize"] = FONT["label"]
    rc["axes.titlesize"] = FONT["title"]
    rc["legend.fontsize"] = FONT["legend"]
    rc["figure.titlesize"] = FONT["suptitle"]
    rc["lines.linewidth"] = LW_REAL
    rc["lines.markersize"] = MS
    rc["legend.framealpha"] = 0.95
    return labels, "labels: English (DejaVu Sans); sizes: ticks {tick}, axis labels {label}, titles {title}, " \
                   "legend {legend} pt".format(**FONT)
