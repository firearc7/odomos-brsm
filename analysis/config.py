"""Paths, plot styling, and shared constants."""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
})

# Seaborn version compatibility
_sns_ver = tuple(int(x) for x in sns.__version__.split(".")[:2])
ERR_KW = {"errorbar": ("ci", 95)} if _sns_ver >= (0, 12) else {"ci": 95}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEMO_CSV = os.path.join(BASE_DIR, "demographic_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.isdir(DATA_DIR):
    DATA_DIR = BASE_DIR

DV_LABELS = {
    "accuracy": "Recognition Accuracy",
    "rt": "Response Time",
    "conf": "Confidence Rating",
}

AB_COLOR = "#E4572E"
NB_COLOR = "#2E86AB"
COND_PALETTE = {"AB": AB_COLOR, "NB": NB_COLOR}
TT_PALETTE = {"EM": "#76B041", "BB": "#F4A259"}
BAR_KW = dict(capsize=0.1, errwidth=1.5, edgecolor="black", linewidth=0.8)
