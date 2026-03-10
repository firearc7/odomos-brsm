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
DATA_DIR = os.path.join(BASE_DIR, "BRSM data csv")
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

# ══════════════════════════════════════════════════════════════════════════════
# 5. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

desc = (
    subj_means.groupby(["condition", "target_type"])
    .agg(
        N=("subject_id", "nunique"),
        Acc_M=("accuracy", "mean"),
        Acc_SD=("accuracy", "std"),
        RT_M=("rt", "mean"),
        RT_SD=("rt", "std"),
        Conf_M=("conf", "mean"),
        Conf_SD=("conf", "std"),
    )
    .reset_index()
)

print("\n  Condition x Target Type (M +/- SD):")
for _, r in desc.iterrows():
    print(
        f"    {r['condition']:>2} x {r['target_type']:>2} (N={int(r['N']):>3}): "
        f"Acc={r['Acc_M']:.3f}+/-{r['Acc_SD']:.3f}  "
        f"RT={r['RT_M']:.3f}+/-{r['RT_SD']:.3f}s  "
        f"Conf={r['Conf_M']:.2f}+/-{r['Conf_SD']:.2f}"
    )

desc.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"), index=False)

# Overall by condition
print("\n  By Condition:")
for cond in ["AB", "NB"]:
    d = subj_means[subj_means["condition"] == cond]
    print(
        f"    {cond}: Acc={d['accuracy'].mean():.3f}+/-{d['accuracy'].std():.3f}  "
        f"RT={d['rt'].mean():.3f}+/-{d['rt'].std():.3f}  "
        f"Conf={d['conf'].mean():.2f}+/-{d['conf'].std():.2f}"
    )

# Overall by target type
print("\n  By Target Type:")
for tt in ["EM", "BB"]:
    d = subj_means[subj_means["target_type"] == tt]
    print(
        f"    {tt}: Acc={d['accuracy'].mean():.3f}+/-{d['accuracy'].std():.3f}  "
        f"RT={d['rt'].mean():.3f}+/-{d['rt'].std():.3f}  "
        f"Conf={d['conf'].mean():.2f}+/-{d['conf'].std():.2f}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# 6. ASSUMPTION CHECKS — Normality + Homogeneity of Variance
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ASSUMPTION CHECKS")
print("=" * 70)

dv_labels = {
    "accuracy": "Recognition Accuracy",
    "rt": "Response Time",
    "conf": "Confidence Rating",
}
normality_ok = {}

for dv in ["accuracy", "rt", "conf"]:
    print(f"\n── {dv_labels[dv]} ──")

    # Shapiro-Wilk on each cell
    all_normal = True
    print("  Shapiro-Wilk test (per cell):")
    for cond in ["AB", "NB"]:
        for tt in ["EM", "BB"]:
            cell = subj_means[
                (subj_means["condition"] == cond) & (subj_means["target_type"] == tt)
            ][dv].dropna()
            if len(cell) >= 3:
                W, p_sw = stats.shapiro(cell)
                tag = "NORMAL" if p_sw >= 0.05 else "NON-NORMAL"
                if p_sw < 0.05:
                    all_normal = False
                print(f"    {cond} x {tt}: W = {W:.4f}, p = {p_sw:.4f} [{tag}]")

    # Levene's test (between-subjects)
    ab_vals = subj_means[subj_means["condition"] == "AB"][dv].dropna()
    nb_vals = subj_means[subj_means["condition"] == "NB"][dv].dropna()
    F_lev, p_lev = stats.levene(ab_vals, nb_vals)
    print(f"  Levene's test (AB vs NB): F = {F_lev:.3f}, p = {p_lev:.4f}")

    normality_ok[dv] = all_normal
    print(f"  => Normality assumption {'MET' if all_normal else 'VIOLATED'}")

# QQ plots (3 DVs x 4 cells)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i, dv in enumerate(["accuracy", "rt", "conf"]):
    for j, (cond, tt) in enumerate(
        [("AB", "EM"), ("AB", "BB"), ("NB", "EM"), ("NB", "BB")]
    ):
        ax = axes[i, j]
        cell = subj_means[
            (subj_means["condition"] == cond) & (subj_means["target_type"] == tt)
        ][dv].dropna()
        stats.probplot(cell, dist="norm", plot=ax)
        ax.set_title(f"{cond} x {tt}\n({dv_labels[dv]})", fontsize=9)
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[0].set_markerfacecolor("steelblue")
fig.suptitle("QQ Plots for Normality Assessment", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_qq_plots.png"))
plt.close()
print("\n  Saved fig5_qq_plots.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. INFERENTIAL STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("INFERENTIAL STATISTICS")
print("=" * 70)

# Ensure each subject has both EM and BB
subj_both = subj_means.groupby("subject_id")["target_type"].nunique()
valid_subjs = subj_both[subj_both == 2].index
anova_data = subj_means[subj_means["subject_id"].isin(valid_subjs)].copy()
print(
    f"\nSubjects with both EM and BB: {len(valid_subjs)} "
    f"(dropped {subj_means['subject_id'].nunique() - len(valid_subjs)})"
)

for dv in ["accuracy", "rt", "conf"]:
    print(f"\n{'─' * 60}")
    print(f"DV: {dv_labels[dv]}")
    print(f"{'─' * 60}")

    dv_data = anova_data.dropna(subset=[dv]).copy()
    subj_check = dv_data.groupby("subject_id")["target_type"].nunique()
    ok_subjs = subj_check[subj_check == 2].index
    dv_data = dv_data[dv_data["subject_id"].isin(ok_subjs)]

    n_ab_dv = dv_data[dv_data["condition"] == "AB"]["subject_id"].nunique()
    n_nb_dv = dv_data[dv_data["condition"] == "NB"]["subject_id"].nunique()
    print(f"  N: AB = {n_ab_dv}, NB = {n_nb_dv}, Total = {n_ab_dv + n_nb_dv}")

    # ── 7a. Parametric: Mixed ANOVA ──
    print("\n  [Parametric] 2x2 Mixed ANOVA:")
    try:
        aov = pg.mixed_anova(
            data=dv_data,
            dv=dv,
            between="condition",
            within="target_type",
            subject="subject_id",
        )
        # Normalize column names (p-unc -> p_unc)
        aov.columns = aov.columns.str.replace("-", "_")
        pc = "p_unc"

        for _, row in aov.iterrows():
            src = row["Source"]
            label = {
                "condition": "Boundary Type",
                "target_type": "Target Type",
                "Interaction": "Interaction",
            }.get(src, src)
            print(
                f"    {label}: F({int(row['DF1'])}, {int(row['DF2'])}) = {row['F']:.3f}, "
                f"p = {row[pc]:.4f}, np2 = {row['np2']:.3f}"
            )

        # Follow-up tests for significant effects
        for _, row in aov.iterrows():
            if row[pc] < 0.05:
                src = row["Source"]

                if src == "condition":
                    print(f"\n    Follow-up (Boundary Type):")
                    ab_m = dv_data[dv_data["condition"] == "AB"].groupby("subject_id")[dv].mean()
                    nb_m = dv_data[dv_data["condition"] == "NB"].groupby("subject_id")[dv].mean()
                    t_val, p_val = stats.ttest_ind(ab_m, nb_m)
                    d_val = pg.compute_effsize(ab_m, nb_m, eftype="cohen")
                    print(
                        f"      AB (M = {ab_m.mean():.3f}) vs NB (M = {nb_m.mean():.3f}): "
                        f"t = {t_val:.3f}, p = {p_val:.4f}, d = {d_val:.3f}"
                    )

                elif src == "target_type":
                    print(f"\n    Follow-up (Target Type):")
                    em_s = dv_data[dv_data["target_type"] == "EM"].set_index("subject_id")[dv]
                    bb_s = dv_data[dv_data["target_type"] == "BB"].set_index("subject_id")[dv]
                    common = em_s.index.intersection(bb_s.index)
                    t_val, p_val = stats.ttest_rel(em_s[common], bb_s[common])
                    d_val = pg.compute_effsize(
                        em_s[common], bb_s[common], paired=True, eftype="cohen"
                    )
                    print(
                        f"      EM (M = {em_s.mean():.3f}) vs BB (M = {bb_s.mean():.3f}): "
                        f"t = {t_val:.3f}, p = {p_val:.4f}, d = {d_val:.3f}"
                    )

                elif src == "Interaction":
                    print(f"\n    Simple effects (Interaction):")
                    # Condition at each target type
                    for tt in ["EM", "BB"]:
                        s = dv_data[dv_data["target_type"] == tt]
                        ab_v = s[s["condition"] == "AB"][dv]
                        nb_v = s[s["condition"] == "NB"][dv]
                        t_val, p_val = stats.ttest_ind(ab_v, nb_v)
                        d_val = pg.compute_effsize(ab_v, nb_v, eftype="cohen")
                        print(
                            f"      {tt}: AB vs NB — t = {t_val:.3f}, "
                            f"p = {p_val:.4f}, d = {d_val:.3f}"
                        )
                    # Target type at each condition
                    for cond_lbl in ["AB", "NB"]:
                        s = dv_data[dv_data["condition"] == cond_lbl]
                        em_v = s[s["target_type"] == "EM"].set_index("subject_id")[dv]
                        bb_v = s[s["target_type"] == "BB"].set_index("subject_id")[dv]
                        common = em_v.index.intersection(bb_v.index)
                        if len(common) > 0:
                            t_val, p_val = stats.ttest_rel(em_v[common], bb_v[common])
                            d_val = pg.compute_effsize(
                                em_v[common], bb_v[common], paired=True, eftype="cohen"
                            )
                            print(
                                f"      {cond_lbl}: EM vs BB — t = {t_val:.3f}, "
                                f"p = {p_val:.4f}, d = {d_val:.3f}"
                            )

    except Exception as e:
        print(f"    ERROR running ANOVA: {e}")

    # ── 7b. Non-parametric alternatives (if normality violated) ──
    if not normality_ok[dv]:
        print(f"\n  [Non-parametric] Normality violated — running rank-based tests:")

        # Between-subjects: Mann-Whitney U on subject-level means
        ab_means = dv_data[dv_data["condition"] == "AB"].groupby("subject_id")[dv].mean()
        nb_means = dv_data[dv_data["condition"] == "NB"].groupby("subject_id")[dv].mean()
        try:
            mwu = pg.mwu(ab_means, nb_means, alternative="two-sided")
            mwu_pc = _pcol(mwu)
            U_val = mwu[_ucol(mwu)].values[0]
            p_mw = mwu[mwu_pc].values[0]
            rbc = mwu["RBC"].values[0]
            print(
                f"    Boundary Type (Mann-Whitney): U = {U_val:.1f}, "
                f"p = {p_mw:.4f}, rank-biserial r = {rbc:.3f}"
            )
        except Exception as e:
            print(f"    Mann-Whitney error: {e}")

        # Within-subjects: Wilcoxon signed-rank on EM - BB difference
        wide = dv_data.pivot_table(index="subject_id", columns="target_type", values=dv)
        diff_scores = (wide["EM"] - wide["BB"]).dropna()
        if len(diff_scores) > 0:
            try:
                wil = pg.wilcoxon(
                    wide.loc[diff_scores.index, "EM"],
                    wide.loc[diff_scores.index, "BB"],
                    alternative="two-sided",
                )
                wil_pc = _pcol(wil)
                W_val = wil[_wcol(wil)].values[0]
                p_wil = wil[wil_pc].values[0]
                rbc_w = wil["RBC"].values[0]
                print(
                    f"    Target Type (Wilcoxon): W = {W_val:.1f}, "
                    f"p = {p_wil:.4f}, rank-biserial r = {rbc_w:.3f}"
                )
            except Exception as e:
                print(f"    Wilcoxon error: {e}")

        # Interaction: Mann-Whitney on difference scores between conditions
        cond_map = (
            dv_data.drop_duplicates("subject_id")
            .set_index("subject_id")["condition"]
        )
        ab_diff = diff_scores[diff_scores.index.isin(cond_map[cond_map == "AB"].index)]
        nb_diff = diff_scores[diff_scores.index.isin(cond_map[cond_map == "NB"].index)]
        if len(ab_diff) > 0 and len(nb_diff) > 0:
            try:
                mwu_int = pg.mwu(ab_diff, nb_diff, alternative="two-sided")
                mwu_int_pc = _pcol(mwu_int)
                U_int = mwu_int[_ucol(mwu_int)].values[0]
                p_int = mwu_int[mwu_int_pc].values[0]
                rbc_int = mwu_int["RBC"].values[0]
                print(
                    f"    Interaction (Mann-Whitney on EM-BB diff): U = {U_int:.1f}, "
                    f"p = {p_int:.4f}, rank-biserial r = {rbc_int:.3f}"
                )

                # Simple effects if interaction significant
                if p_int < 0.05:
                    print("    Non-parametric simple effects:")
                    for tt in ["EM", "BB"]:
                        s = dv_data[dv_data["target_type"] == tt]
                        ab_v = s[s["condition"] == "AB"][dv]
                        nb_v = s[s["condition"] == "NB"][dv]
                        mw_s = pg.mwu(ab_v, nb_v, alternative="two-sided")
                        mw_s_pc = _pcol(mw_s)
                        print(
                            f"      {tt}: AB vs NB — U = {mw_s[_ucol(mw_s)].values[0]:.1f}, "
                            f"p = {mw_s[mw_s_pc].values[0]:.4f}"
                        )
                    for cond_lbl in ["AB", "NB"]:
                        s = dv_data[dv_data["condition"] == cond_lbl]
                        em_v = s[s["target_type"] == "EM"].set_index("subject_id")[dv]
                        bb_v = s[s["target_type"] == "BB"].set_index("subject_id")[dv]
                        common = em_v.index.intersection(bb_v.index)
                        if len(common) > 0:
                            wil_s = pg.wilcoxon(
                                em_v[common], bb_v[common], alternative="two-sided"
                            )
                            wil_s_pc = _pcol(wil_s)
                            print(
                                f"      {cond_lbl}: EM vs BB — W = {wil_s[_wcol(wil_s)].values[0]:.1f}, "
                                f"p = {wil_s[wil_s_pc].values[0]:.4f}"
                            )
            except Exception as e:
                print(f"    Interaction test error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SUPPLEMENTARY: Correlation between DVs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CORRELATIONS BETWEEN DVs (per-subject level)")
print("=" * 70)

# Compute per-subject overall means
subj_overall = (
    trials.groupby(["subject_id", "condition"])
    .agg(accuracy=("accuracy", "mean"), rt=("rt", "mean"), conf=("conf", "mean"))
    .reset_index()
)

for dv_x, dv_y, label in [
    ("accuracy", "rt", "Accuracy vs RT"),
    ("accuracy", "conf", "Accuracy vs Confidence"),
    ("rt", "conf", "RT vs Confidence"),
]:
    valid_corr = subj_overall[[dv_x, dv_y]].dropna()
    if len(valid_corr) >= 3:
        rho, p_rho = stats.spearmanr(valid_corr[dv_x], valid_corr[dv_y])
        print(f"  {label}: Spearman rho = {rho:.3f}, p = {p_rho:.4f} (N = {len(valid_corr)})")

# ══════════════════════════════════════════════════════════════════════════════
# 9. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

AB_COLOR = "#E4572E"
NB_COLOR = "#2E86AB"
cond_palette = {"AB": AB_COLOR, "NB": NB_COLOR}
tt_palette = {"EM": "#76B041", "BB": "#F4A259"}
bar_kw = dict(capsize=0.1, errwidth=1.5, edgecolor="black", linewidth=0.8)

# ── Figure 1: Accuracy Bar Plot ──
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    data=subj_means, x="target_type", y="accuracy", hue="condition",
    hue_order=["AB", "NB"], order=["EM", "BB"],
    palette=cond_palette, **_err_kw, **bar_kw, ax=ax,
)
ax.set_xlabel("Target Frame Type")
ax.set_ylabel("Mean Recognition Accuracy")
ax.set_title("Recognition Accuracy by Condition and Target Type")
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
ax.legend(title="Boundary Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_accuracy_barplot.png"))
plt.close()
print("  Saved fig1_accuracy_barplot.png")

# ── Figure 2: RT Bar Plot ──
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    data=subj_means, x="target_type", y="rt", hue="condition",
    hue_order=["AB", "NB"], order=["EM", "BB"],
    palette=cond_palette, **_err_kw, **bar_kw, ax=ax,
)
ax.set_xlabel("Target Frame Type")
ax.set_ylabel("Mean Response Time (s)")
ax.set_title("Response Time by Condition and Target Type")
ax.legend(title="Boundary Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_rt_barplot.png"))
plt.close()
print("  Saved fig2_rt_barplot.png")

# ── Figure 3: Confidence Bar Plot ──
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    data=subj_means, x="target_type", y="conf", hue="condition",
    hue_order=["AB", "NB"], order=["EM", "BB"],
    palette=cond_palette, **_err_kw, **bar_kw, ax=ax,
)
ax.set_xlabel("Target Frame Type")
ax.set_ylabel("Mean Confidence Rating (1-5)")
ax.set_title("Confidence Rating by Condition and Target Type")
ax.legend(title="Boundary Type")
ax.set_ylim(1, 5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_confidence_barplot.png"))
plt.close()
print("  Saved fig3_confidence_barplot.png")

# ── Figure 4: Accuracy Distribution (Violin + Strip) ──
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(
    data=subj_means, x="condition", y="accuracy", hue="target_type",
    hue_order=["EM", "BB"], order=["AB", "NB"],
    palette=tt_palette, split=True, inner="quartile", alpha=0.7, ax=ax,
)
ax.set_xlabel("Boundary Condition")
ax.set_ylabel("Mean Recognition Accuracy (per subject)")
ax.set_title("Distribution of Recognition Accuracy")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.legend(title="Target Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_accuracy_violin.png"))
plt.close()
print("  Saved fig4_accuracy_violin.png")

# ── Figure 5: QQ Plots — already saved above ──

# ── Figure 8: Task Paradigm Diagram ──
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis("off")

import matplotlib.patches as mpatches

box_kw = dict(boxstyle="round,pad=0.4", facecolor="#E8F0FE", edgecolor="#333", linewidth=1.5)
arrow_kw = dict(arrowstyle="->,head_width=0.3,head_length=0.2", color="#333", linewidth=2)

# Title
ax.text(7, 5.7, "Experimental Paradigm", ha="center", va="center", fontsize=16, fontweight="bold")

# Phase labels
ax.text(3.5, 5.1, "Encoding Phase", ha="center", va="center", fontsize=13, fontweight="bold", color="#2E86AB")
ax.text(10.5, 5.1, "Recognition Phase (2AFC)", ha="center", va="center", fontsize=13, fontweight="bold", color="#E4572E")

# Encoding: Movie clip box
ax.text(1.8, 3.8, "Watch 40\nMovie Clips", ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#D4EDDA", edgecolor="#333", linewidth=1.5))

# Arrow to boundary
ax.annotate("", xy=(3.5, 3.8), xytext=(2.8, 3.8), arrowprops=arrow_kw)

# Boundary box (split)
ax.text(5.0, 4.3, "AB Group:\nAbrupt hard cuts", ha="center", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FDDEDE", edgecolor="#E4572E", linewidth=1.2))
ax.text(5.0, 3.2, "NB Group:\nSmooth transitions", ha="center", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#DEE8F5", edgecolor="#2E86AB", linewidth=1.2))

# Arrow to recognition
ax.annotate("", xy=(7.2, 3.8), xytext=(6.2, 3.8), arrowprops=arrow_kw)
ax.text(6.7, 4.15, "delay", ha="center", va="center", fontsize=9, fontstyle="italic", color="#666")

# Recognition: 2AFC
ax.text(8.5, 3.8, "Target vs. Lure\n(which frame\ndid you see?)", ha="center", va="center", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor="#333", linewidth=1.5))

# Arrow to confidence
ax.annotate("", xy=(10.3, 3.8), xytext=(9.7, 3.8), arrowprops=arrow_kw)

# Confidence
ax.text(11.5, 3.8, "Confidence\nRating\n(1 to 5)", ha="center", va="center", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0E6FF", edgecolor="#333", linewidth=1.5))

# Arrow: repeat
ax.annotate("", xy=(8.5, 2.7), xytext=(11.5, 2.7),
            arrowprops=dict(arrowstyle="<-", color="#999", linewidth=1.5, linestyle="--"))
ax.text(10.0, 2.35, "x 40 trials", ha="center", va="center", fontsize=9, fontstyle="italic", color="#666")

# Target types
ax.text(8.5, 1.5, "Target Frame Types", ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(6.5, 0.8, "Event-Model (EM)\nConsistent with\nevent representation", ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F5E9", edgecolor="#76B041", linewidth=1.2))
ax.text(10.5, 0.8, "Boundary-Break (BB)\nNear an\nevent boundary", ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF0E0", edgecolor="#F4A259", linewidth=1.2))

# DVs
ax.text(13.0, 3.8, "DVs:\nAccuracy\nRT\nConfidence", ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#333", linewidth=1.2))
ax.annotate("", xy=(12.2, 3.8), xytext=(12.6, 3.8),
            arrowprops=dict(arrowstyle="<-", color="#333", linewidth=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig8_task_paradigm.png"))
plt.close()
print("  Saved fig8_task_paradigm.png")

# ── Figure 6: Overall Accuracy Histogram ──
overall_acc = trials.groupby("subject_id")["accuracy"].mean()
condition_map = trials.drop_duplicates("subject_id").set_index("subject_id")["condition"]
fig, ax = plt.subplots(figsize=(8, 5))
for cond, color in [("AB", AB_COLOR), ("NB", NB_COLOR)]:
    ids = condition_map[condition_map == cond].index
    vals = overall_acc[overall_acc.index.isin(ids)]
    ax.hist(
        vals, bins=15, alpha=0.6, color=color, edgecolor="black",
        linewidth=0.5, label=f"{cond} (N={len(vals)})",
    )
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")
ax.set_xlabel("Overall Recognition Accuracy")
ax.set_ylabel("Number of Subjects")
ax.set_title("Distribution of Subject-Level Accuracy")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_accuracy_histogram.png"))
plt.close()
print("  Saved fig6_accuracy_histogram.png")

# ── Figure 7: Confidence Interaction Plot ──
fig, ax = plt.subplots(figsize=(8, 6))
interaction = (
    subj_means.groupby(["condition", "target_type"])
    .agg(
        conf_mean=("conf", "mean"),
        conf_se=("conf", lambda x: x.std() / np.sqrt(len(x))),
    )
    .reset_index()
)
for cond, color, marker in [("AB", AB_COLOR, "s"), ("NB", NB_COLOR, "o")]:
    d = interaction[interaction["condition"] == cond]
    ax.errorbar(
        d["target_type"], d["conf_mean"], yerr=d["conf_se"],
        marker=marker, markersize=10, linewidth=2.5, capsize=5,
        color=color, label=f"{cond} ({'Abrupt' if cond == 'AB' else 'Natural'})",
    )
ax.set_xlabel("Target Frame Type")
ax.set_ylabel("Mean Confidence Rating (+/- SE)")
ax.set_title("Interaction: Boundary Type x Target Frame Type (Confidence)")
ax.legend(title="Boundary Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_confidence_interaction.png"))
plt.close()
print("  Saved fig7_confidence_interaction.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
subj_means.to_csv(os.path.join(OUTPUT_DIR, "subject_means.csv"), index=False)
trials.to_csv(os.path.join(OUTPUT_DIR, "all_trials_clean.csv"), index=False)
demographics.to_csv(os.path.join(OUTPUT_DIR, "demographics.csv"), index=False)

print(f"\n{'=' * 70}")
print("OUTPUT FILES SAVED:")
print(f"{'=' * 70}")
for f in [
    "descriptive_statistics.csv", "subject_means.csv",
    "all_trials_clean.csv", "demographics.csv",
    "fig1_accuracy_barplot.png", "fig2_rt_barplot.png",
    "fig3_confidence_barplot.png", "fig4_accuracy_violin.png",
    "fig5_qq_plots.png", "fig6_accuracy_histogram.png",
    "fig7_confidence_interaction.png", "fig8_task_paradigm.png",
]:
    print(f"  output/{f}")
print("\nDone!")
