"""
Movie Memory Experiment — Preliminary Data Analysis
=====================================================
Team Odomos
Archit Choudhary (2023114002), Bhavya Ahuja (2023111035), Hrishiraj Mitra (2023111037)

Design: 2 (Boundary Type: AB vs NB; between) x 2 (Target Type: EM vs BB; within)
DVs: Recognition Accuracy, Response Time, Confidence Rating

Generates: descriptive statistics, normality checks, parametric/non-parametric
inferential tests, and publication-quality figures.
"""

import os
import re
import glob
import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats

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
_err_kw = {"errorbar": ("ci", 95)} if _sns_ver >= (0, 12) else {"ci": 95}

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.isdir(DATA_DIR):
    DATA_DIR = BASE_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def parse_rt(val):
    """Parse RT from PsychoPy format — float, string list, or NaN."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s in ("None", "", "[]"):
        return np.nan
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return float(parsed[0]) if len(parsed) > 0 else np.nan
        return float(parsed)
    except Exception:
        try:
            return float(s)
        except ValueError:
            return np.nan


def extract_target_type(target_img):
    """Extract EM or BB from target_img filename."""
    if pd.isna(target_img):
        return np.nan
    s = str(target_img)
    if "_EM_T" in s:
        return "EM"
    elif "_BB_T" in s:
        return "BB"
    return np.nan


def extract_condition(filename):
    """Extract AB or NB from filename (handles spaces)."""
    fn = os.path.basename(filename).upper().replace(" ", "")
    if "_AB_" in fn:
        return "AB"
    elif "_NB_" in fn:
        return "NB"
    return None


def extract_subject_id(filename):
    """Extract subject ID from filename."""
    fn = os.path.basename(filename)
    m = re.match(r"(sub\d+)", fn, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _pcol(df):
    """Return the p-value column name in a pingouin result DataFrame."""
    for c in ["p_val", "p-val", "p_unc", "p-unc"]:
        if c in df.columns:
            return c
    return "p_val"


def _ucol(df):
    """Return the U-value column name."""
    for c in ["U_val", "U-val"]:
        if c in df.columns:
            return c
    return "U_val"


def _wcol(df):
    """Return the W/T-value column name."""
    for c in ["W_val", "W-val", "T_val", "T-val"]:
        if c in df.columns:
            return c
    return "W_val"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("MOVIE MEMORY EXPERIMENT — PRELIMINARY ANALYSIS")
print("Team Odomos")
print("=" * 70)

csv_files = (
    glob.glob(os.path.join(DATA_DIR, "sub*_recognitionstage_*.csv"))
    + glob.glob(os.path.join(DATA_DIR, "Sub*_recognitionstage_*.csv"))
    + glob.glob(os.path.join(DATA_DIR, "sub*_AB *_recognitionstage_*.csv"))
    + glob.glob(os.path.join(DATA_DIR, "sub*_NB *_recognitionstage_*.csv"))
    + glob.glob(os.path.join(BASE_DIR, "sub*_recognitionstage_*.csv"))
    + glob.glob(os.path.join(BASE_DIR, "Sub*_recognitionstage_*.csv"))
)
csv_files = list(set(csv_files))

print(f"\nFound {len(csv_files)} subject CSV files.")

all_trials = []
demographics_list = []
vigilance_list = []
skipped = []

for fpath in csv_files:
    try:
        df = pd.read_csv(fpath, low_memory=False)
    except Exception as e:
        skipped.append((fpath, str(e)))
        continue

    condition = extract_condition(fpath)
    subject_id = extract_subject_id(fpath)
    if condition is None or subject_id is None:
        skipped.append((fpath, "Could not determine condition or subject ID"))
        continue

    # Demographics
    demo_row = {"subject_id": subject_id, "condition": condition}
    if len(df) > 1:
        for col in ["age", "gender", "handedness", "vision",
                     "caffeine_2h", "alcohol_smoke_12h"]:
            if col in df.columns:
                demo_row[col] = df[col].iloc[0]
    demographics_list.append(demo_row)

    # Recognition trials: rows where movie_id is not empty
    if "movie_id" not in df.columns:
        skipped.append((fpath, "No movie_id column"))
        continue

    recog = df[df["movie_id"].notna() & (df["movie_id"] != "")].copy()
    if len(recog) == 0:
        skipped.append((fpath, "No recognition trials"))
        continue

    recog["movie_id"] = pd.to_numeric(recog["movie_id"], errors="coerce")
    recog = recog[recog["movie_id"].notna()].copy()

    # Accuracy
    if "resp.corr" in recog.columns:
        recog["accuracy"] = pd.to_numeric(recog["resp.corr"], errors="coerce")
    if "recogloop.resp.corr" in recog.columns:
        alt_acc = pd.to_numeric(recog["recogloop.resp.corr"], errors="coerce")
        if "accuracy" in recog.columns:
            recog["accuracy"] = recog["accuracy"].fillna(alt_acc)
        else:
            recog["accuracy"] = alt_acc

    # RT
    for rt_col in ["resp.rt", "recogloop.resp.rt"]:
        if rt_col in recog.columns:
            parsed = recog[rt_col].apply(parse_rt)
            if "rt" not in recog.columns:
                recog["rt"] = parsed
            else:
                recog["rt"] = recog["rt"].fillna(parsed)

    # Confidence
    for conf_col in ["conf_radio.response", "recogloop.conf_radio.response", "confidence"]:
        if conf_col in recog.columns:
            parsed = pd.to_numeric(recog[conf_col], errors="coerce")
            if "conf" not in recog.columns:
                recog["conf"] = parsed
            else:
                recog["conf"] = recog["conf"].fillna(parsed)

    # Target type from target_img
    if "target_img" in recog.columns:
        recog["target_type"] = recog["target_img"].apply(extract_target_type)
    else:
        skipped.append((fpath, "No target_img column"))
        continue

    valid = recog[recog["target_type"].notna()].copy()
    if len(valid) == 0:
        skipped.append((fpath, "No valid target types"))
        continue

    valid["subject_id"] = subject_id
    valid["condition"] = condition
    all_trials.append(
        valid[["subject_id", "condition", "movie_id",
               "target_type", "accuracy", "rt", "conf"]].copy()
    )

    # Vigilance (AB only)
    if condition == "AB" and "vigilance_correct" in df.columns:
        movie_rows = df[df["path"].notna() & (df["path"] != "")].copy()
        if "is_repeat" in movie_rows.columns:
            repeats = movie_rows[
                pd.to_numeric(movie_rows["is_repeat"], errors="coerce") == 1
            ]
            if len(repeats) > 0:
                vig_correct = pd.to_numeric(
                    repeats["vigilance_correct"], errors="coerce"
                )
                vigilance_list.append({
                    "subject_id": subject_id,
                    "condition": condition,
                    "vigilance_hit_rate": vig_correct.mean(),
                    "n_repeats": len(repeats),
                })

if skipped:
    print(f"\nSkipped {len(skipped)} files:")
    for fp, reason in skipped[:10]:
        print(f"  {os.path.basename(fp)}: {reason}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")

# ══════════════════════════════════════════════════════════════════════════════
# 2. COMBINE DATA
# ══════════════════════════════════════════════════════════════════════════════
trials = pd.concat(all_trials, ignore_index=True)
demographics = pd.DataFrame(demographics_list)
vigilance_df = pd.DataFrame(vigilance_list) if vigilance_list else pd.DataFrame()

n_subj = trials["subject_id"].nunique()
n_ab = trials[trials["condition"] == "AB"]["subject_id"].nunique()
n_nb = trials[trials["condition"] == "NB"]["subject_id"].nunique()
print(f"\nTotal trials: {len(trials)}")
print(f"Subjects: {n_subj} (AB = {n_ab}, NB = {n_nb})")
print(f"Trials per subject: ~{len(trials) / n_subj:.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

n0 = len(trials)
trials = trials.dropna(subset=["accuracy"])
print(f"  Dropped {n0 - len(trials)} rows with missing accuracy.")

# Subject-level accuracy check
subj_acc = trials.groupby("subject_id")["accuracy"].mean()
low_acc = subj_acc[subj_acc < 0.55]
print(f"  Subjects below 55% accuracy (near chance): {len(low_acc)}")
if len(low_acc) > 0:
    print(f"    IDs: {', '.join(low_acc.index.tolist())}")

# Vigilance check
if len(vigilance_df) > 0:
    low_vig = vigilance_df[vigilance_df["vigilance_hit_rate"] < 0.5]
    print(f"  AB subjects with <50% vigilance hit rate: {len(low_vig)}")

# RT outlier removal
n_rt = trials["rt"].notna().sum()
trials.loc[(trials["rt"] < 0.2) | (trials["rt"] > 60), "rt"] = np.nan
n_rt_removed = n_rt - trials["rt"].notna().sum()
print(f"  RT outliers removed (< 0.2 s or > 60 s): {n_rt_removed}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PER-SUBJECT MEANS
# ══════════════════════════════════════════════════════════════════════════════
subj_means = (
    trials.groupby(["subject_id", "condition", "target_type"])
    .agg(
        accuracy=("accuracy", "mean"),
        rt=("rt", "mean"),
        conf=("conf", "mean"),
        n_trials=("accuracy", "count"),
    )
    .reset_index()
)
