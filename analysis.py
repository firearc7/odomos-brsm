"""
Movie Memory Experiment — Full Data Analysis
=============================================
Team Odomos
Archit Choudhary (2023114002), Bhavya Ahuja (2023111035), Hrishiraj Mitra (2023111037)

Design: 2 (Boundary Type: AB vs NB; between) x 2 (Target Type: EM vs BB; within)
DVs: Recognition Accuracy, Response Time, Confidence Rating

Hypotheses:
  H1: NB > AB in recognition accuracy (boundary type main effect)
  H2: BB > EM in recognition accuracy (boundary advantage; Radvansky & Zacks, 2017)
  H3: Condition x Target Type interaction on accuracy
  H4: Condition x Target Type interaction on RT (AB slows for BB)
  H5: NB shows higher d' (signal detection) than AB
  H6: Demographics (age, gender, vision) moderate recognition accuracy

Analyses: descriptive statistics, normality checks, parametric/non-parametric
tests, Signal Detection Theory, mixed-effects models with crossed random
effects, demographic moderation, and publication-quality figures.
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
import statsmodels.formula.api as smf
import statsmodels.api as sm

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
print("MOVIE MEMORY EXPERIMENT — FULL ANALYSIS")
print("Team Odomos")
print("=" * 70)

DEMO_CSV = os.path.join(BASE_DIR, "demographic_data.csv")

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

# ── Figure 1: Accuracy Interaction Plot (with individual data) ──
fig, ax = plt.subplots(figsize=(8, 6))
acc_interaction = (
    subj_means.dropna(subset=["accuracy"])
    .groupby(["condition", "target_type"])
    .agg(acc_mean=("accuracy", "mean"), acc_se=("accuracy", lambda x: x.std() / np.sqrt(len(x))))
    .reset_index()
)
for cond, color, marker in [("AB", AB_COLOR, "s"), ("NB", NB_COLOR, "o")]:
    d = acc_interaction[acc_interaction["condition"] == cond]
    ax.errorbar(
        d["target_type"], d["acc_mean"], yerr=d["acc_se"],
        marker=marker, markersize=10, linewidth=2.5, capsize=5,
        color=color, label=f"{cond} ({'Abrupt' if cond == 'AB' else 'Natural'})",
    )
# Overlay individual data points
for cond, color in [("AB", AB_COLOR), ("NB", NB_COLOR)]:
    d = subj_means[(subj_means["condition"] == cond) & subj_means["accuracy"].notna()]
    x_jitter = np.where(d["target_type"] == "EM", -0.05, 0.05)
    ax.scatter(
        np.where(d["target_type"] == "EM", 0, 1) + x_jitter + np.random.normal(0, 0.02, len(d)),
        d["accuracy"], alpha=0.15, s=15, color=color, zorder=1,
    )
ax.set_xlabel("Target Frame Type")
ax.set_ylabel("Mean Recognition Accuracy (± SE)")
ax.set_title("Accuracy Interaction: Boundary Type × Target Type")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
ax.set_ylim(0.4, 1.0)
ax.legend(title="Boundary Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_accuracy_interaction.png"))
plt.close()
print("  Saved fig1_accuracy_interaction.png")

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
fig, ax = plt.subplots(figsize=(9.35, 5.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title and Headers
ax.text(50, 94, "Experimental Paradigm", ha="center", va="center", fontsize=18, fontweight="bold", color="#1e293b")
ax.text(24, 85, "Phase 1: Encoding", ha="center", va="center", fontsize=15, fontweight="bold", color="#0284c7")
ax.text(76, 85, "Phase 2: Recognition (2AFC)", ha="center", va="center", fontsize=15, fontweight="bold", color="#ea580c")

# Helper for drawing boxes
def draw_box(x, y, text, border_color, fill_color, fontsize=11, style="round,pad=0.6", ha="center"):
    bbox = dict(boxstyle=style, facecolor=fill_color, edgecolor=border_color, linewidth=1.5)
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color="#1e293b", bbox=bbox, zorder=3)

# Encoding block
draw_box(24, 68, "Watch 40 Movie Clips\n(Abrupt vs. Natural Boundaries)", "#3b82f6", "#eff6ff")

# Retention Delay arrow
arrow_kw = dict(arrowstyle="-|>,head_width=0.5,head_length=0.6", color="#475569", linewidth=2.5)
ax.annotate("", xy=(58, 68), xytext=(44, 68), arrowprops=arrow_kw, zorder=2)
ax.text(51, 73, "Retention Delay", ha="center", va="center", fontsize=11, fontstyle="italic", color="#475569")

# Recognition blocks
draw_box(76, 68, "Target Frame vs. Lure Frame", "#f59e0b", "#fef3c7")
ax.annotate("", xy=(76, 56), xytext=(76, 62), arrowprops=arrow_kw, zorder=2)

draw_box(76, 46, "Target Frame Types:\n1. Event-Model (EM)\n2. Boundary-Break (BB)", "#22c55e", "#dcfce7")
ax.annotate("", xy=(76, 33), xytext=(76, 39), arrowprops=arrow_kw, zorder=2)

draw_box(76, 26, "Confidence Rating (1-5)", "#a855f7", "#f3e8ff")

# Repeated Trials Loop (x 40 trials)
loop_kw = dict(arrowstyle="-|>,head_width=0.4,head_length=0.6", color="#64748b", linewidth=2, linestyle="--")
ax.plot([86, 94, 94], [26, 26, 68], color="#64748b", linewidth=2, linestyle="--", zorder=1)
ax.annotate("", xy=(87, 68), xytext=(94, 68), arrowprops=loop_kw, zorder=2)
ax.text(97, 47, "x 40 trials", ha="center", va="center", fontsize=11, fontstyle="italic", color="#64748b", rotation=270)

# Measured variables and Condition setup
draw_box(4, 26, "Measured Variables (DVs):\n- Accuracy (Hit / Miss)\n- Response Time (RT)\n- Confidence Score", 
         "#334155", "#f8fafc", style="square,pad=0.8", ha="left")

draw_box(47, 26, "Condition Setup:\nAB Group: Abrupt hard cuts\nNB Group: Smooth transitions", 
         "#ea580c", "#ffedd5", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig8_task_paradigm.png"), dpi=300, bbox_inches='tight')
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
# 10. SAVE OUTPUTS (original)
# ══════════════════════════════════════════════════════════════════════════════
subj_means.to_csv(os.path.join(OUTPUT_DIR, "subject_means.csv"), index=False)
trials.to_csv(os.path.join(OUTPUT_DIR, "all_trials_clean.csv"), index=False)
demographics.to_csv(os.path.join(OUTPUT_DIR, "demographics.csv"), index=False)

print("\nOriginal outputs saved. Running extended analyses...")


# ══════════════════════════════════════════════════════════════════════════════
# 2b. LOAD AND MERGE EXTERNAL DEMOGRAPHIC DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DEMOGRAPHIC DATA INTEGRATION")
print("=" * 70)

demo_ext = pd.read_csv(DEMO_CSV)
demo_ext.columns = demo_ext.columns.str.strip()
demo_ext = demo_ext.rename(columns={
    "Sub ID": "sub_id_raw", "Age": "age_demo",
    "Gender": "gender_demo", "Handedness": "hand_demo", "Vision": "vision_demo",
})
# Normalise subject ID: lowercase, strip spaces
demo_ext["subject_id"] = (
    demo_ext["sub_id_raw"].astype(str).str.strip().str.lower()
    .str.replace(r"_[an]b$", "", regex=True)
)
# Deduplicate
demo_ext = demo_ext.drop_duplicates(subset="subject_id", keep="first")

# Impute missing demographics using appropriate central tendency
# Age (continuous, slightly right-skewed) → median
age_median = demo_ext["age_demo"].median()
demo_ext["age_demo"] = demo_ext["age_demo"].fillna(age_median)
# Gender, Handedness, Vision (categorical) → mode
for col in ["gender_demo", "hand_demo", "vision_demo"]:
    mode_val = demo_ext[col].mode().iloc[0] if len(demo_ext[col].mode()) > 0 else "Unknown"
    demo_ext[col] = demo_ext[col].fillna(mode_val)

n_imputed = demo_ext["sub_id_raw"].isna().sum()
print(f"  Demographic CSV: {len(demo_ext)} entries")
print(f"  Age imputed with median ({age_median:.0f}) for missing values")
print(f"  Categorical vars imputed with mode for missing values")

# Merge with trials
trials = trials.merge(
    demo_ext[["subject_id", "age_demo", "gender_demo", "hand_demo", "vision_demo"]],
    on="subject_id", how="left",
)
# Fill any remaining NaN from subjects not in demo CSV
trials["age_demo"] = trials["age_demo"].fillna(age_median)
for col in ["gender_demo", "hand_demo", "vision_demo"]:
    mode_val = trials[col].mode().iloc[0] if len(trials[col].mode()) > 0 else "Unknown"
    trials[col] = trials[col].fillna(mode_val)

# Print demographic summary
print("\n  Demographic Summary:")
age_vals = trials.drop_duplicates("subject_id")["age_demo"]
print(f"    Age: M = {age_vals.mean():.1f}, SD = {age_vals.std():.1f}, "
      f"range = {age_vals.min():.0f}-{age_vals.max():.0f}")
gender_counts = trials.drop_duplicates("subject_id")["gender_demo"].value_counts()
for g, c in gender_counts.items():
    print(f"    {g}: {c}")
hand_counts = trials.drop_duplicates("subject_id")["hand_demo"].value_counts()
for h, c in hand_counts.items():
    print(f"    {h}: {c}")
vision_counts = trials.drop_duplicates("subject_id")["vision_demo"].value_counts()
for v, c in vision_counts.items():
    print(f"    {v}: {c}")

# Also merge into subj_means
subj_means = subj_means.merge(
    demo_ext[["subject_id", "age_demo", "gender_demo", "hand_demo", "vision_demo"]],
    on="subject_id", how="left",
)
for col in ["age_demo", "gender_demo", "hand_demo", "vision_demo"]:
    if col == "age_demo":
        subj_means[col] = subj_means[col].fillna(age_median)
    else:
        mode_val = subj_means[col].mode().iloc[0] if len(subj_means[col].mode()) > 0 else "Unknown"
        subj_means[col] = subj_means[col].fillna(mode_val)

# ══════════════════════════════════════════════════════════════════════════════
# 11. SIGNAL DETECTION THEORY (H5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SIGNAL DETECTION THEORY (H5)")
print("=" * 70)

# Compute per-subject hit rate and false alarm rate
# In this 2AFC task: accuracy IS the hit rate effectively.
# For SDT in 2AFC: d' = z(HR) - z(FAR)
# HR = P(correct | target present) = accuracy for target trials
# FAR = P(incorrect | lure present) = 1 - accuracy
# For 2AFC: d' = sqrt(2) * z(proportion correct) [Macmillan & Creelman, 2005]

sdt_list = []
for (sid, cond), grp in trials.groupby(["subject_id", "condition"]):
    for tt in ["EM", "BB"]:
        tt_trials = grp[grp["target_type"] == tt]
        n = len(tt_trials)
        if n == 0:
            continue
        n_correct = tt_trials["accuracy"].sum()
        # Log-linear correction (Hautus, 1995): add 0.5 to hits and misses
        hr = (n_correct + 0.5) / (n + 1)
        far = 1 - hr
        d_prime = stats.norm.ppf(hr) - stats.norm.ppf(far)
        # Note: criterion c is not meaningful in 2AFC (always ~0 by design)
        sdt_list.append({
            "subject_id": sid, "condition": cond, "target_type": tt,
            "hit_rate": hr, "false_alarm_rate": far,
            "d_prime": d_prime, "n_trials": n,
        })

sdt_df = pd.DataFrame(sdt_list)

# Overall d' by condition
print("\n  d' by Condition and Target Type (M ± SD):")
sdt_desc = sdt_df.groupby(["condition", "target_type"]).agg(
    d_M=("d_prime", "mean"), d_SD=("d_prime", "std"),
    N=("subject_id", "nunique"),
).reset_index()
for _, r in sdt_desc.iterrows():
    print(f"    {r['condition']} x {r['target_type']}: "
          f"d' = {r['d_M']:.3f} ± {r['d_SD']:.3f}")

# H5: Compare d' between AB and NB (collapsed across target type)
sdt_subj = sdt_df.groupby(["subject_id", "condition"])["d_prime"].mean().reset_index()
ab_dp = sdt_subj[sdt_subj["condition"] == "AB"]["d_prime"]
nb_dp = sdt_subj[sdt_subj["condition"] == "NB"]["d_prime"]

print("\n  H5: d' comparison (NB > AB?)")
# Normality check
_, p_sw_ab = stats.shapiro(ab_dp)
_, p_sw_nb = stats.shapiro(nb_dp)
print(f"    Shapiro-Wilk: AB p = {p_sw_ab:.4f}, NB p = {p_sw_nb:.4f}")

if p_sw_ab >= 0.05 and p_sw_nb >= 0.05:
    t_val, p_val = stats.ttest_ind(nb_dp, ab_dp)
    d_val = pg.compute_effsize(nb_dp, ab_dp, eftype="cohen")
    print(f"    Independent t-test: t = {t_val:.3f}, p = {p_val:.4f}, d = {d_val:.3f}")
else:
    mwu = pg.mwu(nb_dp, ab_dp, alternative="two-sided")
    pc = _pcol(mwu)
    print(f"    Mann-Whitney U = {mwu[_ucol(mwu)].values[0]:.1f}, "
          f"p = {mwu[pc].values[0]:.4f}, RBC = {mwu['RBC'].values[0]:.3f}")
# Also run t-test regardless for reporting
t_val_dp, p_val_dp = stats.ttest_ind(nb_dp, ab_dp)
d_val_dp = pg.compute_effsize(nb_dp, ab_dp, eftype="cohen")
print(f"    t = {t_val_dp:.3f}, p = {p_val_dp:.4f}, d = {d_val_dp:.3f}")
print(f"    AB d' M = {ab_dp.mean():.3f}, NB d' M = {nb_dp.mean():.3f}")

# Note: Criterion c is not reported — in 2AFC, response bias is undefined
# (participants must choose one of two options, so c ≈ 0 by construction).

# 2x2 ANOVA on d'
sdt_both = sdt_df.copy()
subj_check_sdt = sdt_both.groupby("subject_id")["target_type"].nunique()
ok_sdt = subj_check_sdt[subj_check_sdt == 2].index
sdt_both = sdt_both[sdt_both["subject_id"].isin(ok_sdt)]

print("\n  2x2 Mixed ANOVA on d':")
try:
    aov_dp = pg.mixed_anova(
        data=sdt_both, dv="d_prime", between="condition",
        within="target_type", subject="subject_id",
    )
    aov_dp.columns = aov_dp.columns.str.replace("-", "_")
    for _, row in aov_dp.iterrows():
        src = row["Source"]
        print(f"    {src}: F({int(row['DF1'])}, {int(row['DF2'])}) = {row['F']:.3f}, "
              f"p = {row['p_unc']:.4f}, np2 = {row['np2']:.3f}")
except Exception as e:
    print(f"    ANOVA error: {e}")

sdt_df.to_csv(os.path.join(OUTPUT_DIR, "sdt_results.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 12. MIXED-EFFECTS MODELS WITH CROSSED RANDOM EFFECTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MIXED-EFFECTS MODELS (Crossed Random Effects)")
print("=" * 70)

# Prepare trial-level data with proper coding
me_data = trials.dropna(subset=["accuracy"]).copy()
me_data["cond_code"] = (me_data["condition"] == "NB").astype(float)
me_data["tt_code"] = (me_data["target_type"] == "EM").astype(float)
me_data["cond_x_tt"] = me_data["cond_code"] * me_data["tt_code"]
me_data["movie_id"] = me_data["movie_id"].astype(int).astype(str)

for dv, dv_label in [("accuracy", "Accuracy"), ("rt", "RT"), ("conf", "Confidence")]:
    print(f"\n── Mixed Model: {dv_label} ──")
    dv_trials = me_data.dropna(subset=[dv]).copy()

    try:
        # Model with random intercepts for subjects (primary) and items
        # statsmodels MixedLM: use subject as groups, add movie_id via vc_formula
        vc = {"movie_id": "0 + C(movie_id)"}
        model = smf.mixedlm(
            f"{dv} ~ cond_code * tt_code",
            data=dv_trials, groups=dv_trials["subject_id"],
            vc_formula=vc,
        )
        result = model.fit(reml=True, method="lbfgs")
        print(result.summary().tables[1].to_string())

        # Extract key fixed effects
        for param in ["cond_code", "tt_code", "cond_code:tt_code"]:
            if param in result.params.index:
                coef = result.params[param]
                se = result.bse[param]
                z = result.tvalues[param]
                p = result.pvalues[param]
                print(f"    {param}: b = {coef:.4f}, SE = {se:.4f}, z = {z:.3f}, p = {p:.4f}")

        # Random effects variance
        print(f"    Subject RE variance: {result.cov_re.iloc[0, 0]:.4f}")

    except Exception as e:
        print(f"    Model failed: {e}")
        # Fallback: simpler model without crossed RE
        try:
            model_simple = smf.mixedlm(
                f"{dv} ~ cond_code * tt_code",
                data=dv_trials, groups=dv_trials["subject_id"],
            )
            result_simple = model_simple.fit(reml=True)
            print("    (Fallback: subject-only random intercept)")
            print(result_simple.summary().tables[1].to_string())
        except Exception as e2:
            print(f"    Fallback also failed: {e2}")

# ══════════════════════════════════════════════════════════════════════════════
# 13. RT INTERACTION DEEP-DIVE (H4)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RT INTERACTION ANALYSIS (H4)")
print("=" * 70)

rt_data = anova_data.dropna(subset=["rt"]).copy()
rt_check = rt_data.groupby("subject_id")["target_type"].nunique()
rt_ok = rt_check[rt_check == 2].index
rt_data = rt_data[rt_data["subject_id"].isin(rt_ok)]

# Simple effects for RT
print("\n  Simple effects — RT:")
for cond_lbl in ["AB", "NB"]:
    s = rt_data[rt_data["condition"] == cond_lbl]
    em_v = s[s["target_type"] == "EM"].set_index("subject_id")["rt"]
    bb_v = s[s["target_type"] == "BB"].set_index("subject_id")["rt"]
    common = em_v.index.intersection(bb_v.index)
    if len(common) > 0:
        t_val, p_val = stats.ttest_rel(em_v[common], bb_v[common])
        d_val = pg.compute_effsize(em_v[common], bb_v[common], paired=True, eftype="cohen")
        print(f"    {cond_lbl}: EM (M={em_v.mean():.3f}) vs BB (M={bb_v.mean():.3f}) — "
              f"t = {t_val:.3f}, p = {p_val:.4f}, d = {d_val:.3f}")

# Between-subjects at each target type
for tt in ["EM", "BB"]:
    s = rt_data[rt_data["target_type"] == tt]
    ab_v = s[s["condition"] == "AB"]["rt"]
    nb_v = s[s["condition"] == "NB"]["rt"]
    t_val, p_val = stats.ttest_ind(ab_v, nb_v)
    d_val = pg.compute_effsize(ab_v, nb_v, eftype="cohen")
    print(f"    {tt}: AB (M={ab_v.mean():.3f}) vs NB (M={nb_v.mean():.3f}) — "
          f"t = {t_val:.3f}, p = {p_val:.4f}, d = {d_val:.3f}")

# Bayesian t-test on the interaction (EM-BB difference scores)
rt_wide = rt_data.pivot_table(index="subject_id", columns="target_type", values="rt")
rt_diff = (rt_wide["EM"] - rt_wide["BB"]).dropna()
cond_map_rt = rt_data.drop_duplicates("subject_id").set_index("subject_id")["condition"]
ab_rt_diff = rt_diff[rt_diff.index.isin(cond_map_rt[cond_map_rt == "AB"].index)]
nb_rt_diff = rt_diff[rt_diff.index.isin(cond_map_rt[cond_map_rt == "NB"].index)]

print(f"\n  RT difference scores (EM - BB):")
print(f"    AB: M = {ab_rt_diff.mean():.3f}, SD = {ab_rt_diff.std():.3f}")
print(f"    NB: M = {nb_rt_diff.mean():.3f}, SD = {nb_rt_diff.std():.3f}")

# Bayesian analysis using pingouin
try:
    bf_rt = pg.bayesfactor_ttest(
        stats.ttest_ind(ab_rt_diff, nb_rt_diff)[0],
        len(ab_rt_diff), len(nb_rt_diff),
    )
    print(f"    Bayes Factor (BF10): {bf_rt:.3f}")
    if bf_rt < 1/3:
        print("    => Moderate evidence for H0 (no interaction)")
    elif bf_rt < 1:
        print("    => Anecdotal evidence for H0")
    elif bf_rt < 3:
        print("    => Anecdotal evidence for H1 (interaction)")
    else:
        print("    => Moderate+ evidence for H1 (interaction)")
except Exception as e:
    print(f"    Bayes Factor error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 14. DEMOGRAPHIC MODERATION ANALYSIS (H6)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DEMOGRAPHIC MODERATION (H6)")
print("=" * 70)

# Per-subject overall accuracy with demographics
subj_demo = (
    trials.groupby(["subject_id", "condition", "age_demo", "gender_demo",
                     "hand_demo", "vision_demo"])
    .agg(accuracy=("accuracy", "mean"), rt=("rt", "mean"), conf=("conf", "mean"))
    .reset_index()
)

# Correlation of age with DVs
print("\n  Age correlations:")
for dv in ["accuracy", "rt", "conf"]:
    valid = subj_demo[["age_demo", dv]].dropna()
    if len(valid) >= 3:
        rho, p_rho = stats.spearmanr(valid["age_demo"], valid[dv])
        print(f"    Age vs {dv}: rho = {rho:.3f}, p = {p_rho:.4f}")

# Gender effect on accuracy
print("\n  Gender effect on accuracy:")
for g in subj_demo["gender_demo"].unique():
    g_data = subj_demo[subj_demo["gender_demo"] == g]["accuracy"]
    print(f"    {g}: M = {g_data.mean():.3f}, SD = {g_data.std():.3f}, N = {len(g_data)}")

males = subj_demo[subj_demo["gender_demo"] == "Male"]["accuracy"]
females = subj_demo[subj_demo["gender_demo"] == "Female"]["accuracy"]
if len(males) >= 3 and len(females) >= 3:
    t_g, p_g = stats.ttest_ind(males, females)
    d_g = pg.compute_effsize(males, females, eftype="cohen")
    print(f"    t = {t_g:.3f}, p = {p_g:.4f}, d = {d_g:.3f}")

# Vision effect on accuracy
print("\n  Vision correction effect on accuracy:")
for v in subj_demo["vision_demo"].unique():
    v_data = subj_demo[subj_demo["vision_demo"] == v]["accuracy"]
    print(f"    {v}: M = {v_data.mean():.3f}, SD = {v_data.std():.3f}, N = {len(v_data)}")

normal_vis = subj_demo[subj_demo["vision_demo"] == "Normal"]["accuracy"]
corrected_vis = subj_demo[subj_demo["vision_demo"] == "Corrected to normal"]["accuracy"]
if len(normal_vis) >= 3 and len(corrected_vis) >= 3:
    t_v, p_v = stats.ttest_ind(normal_vis, corrected_vis)
    d_v = pg.compute_effsize(normal_vis, corrected_vis, eftype="cohen")
    print(f"    Normal vs Corrected: t = {t_v:.3f}, p = {p_v:.4f}, d = {d_v:.3f}")

# ANCOVA: accuracy ~ condition * target_type + age + gender
print("\n  ANCOVA (accuracy ~ condition + age_demo + gender_demo + vision_demo):")
subj_demo["gender_code"] = (subj_demo["gender_demo"] == "Male").astype(float)
subj_demo["vision_code"] = (subj_demo["vision_demo"] == "Corrected to normal").astype(float)
try:
    ancova_result = pg.ancova(
        data=subj_demo, dv="accuracy", between="condition",
        covar=["age_demo", "gender_code", "vision_code"],
    )
    print(ancova_result.to_string())
except Exception as e:
    print(f"    ANCOVA error: {e}")
    # Fallback: OLS
    try:
        ols_model = smf.ols(
            "accuracy ~ C(condition) + age_demo + gender_code + vision_code",
            data=subj_demo,
        ).fit()
        print(ols_model.summary2().tables[1].to_string())
    except Exception as e2:
        print(f"    OLS error: {e2}")

# ══════════════════════════════════════════════════════════════════════════════
# 15. ADDITIONAL VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ADDITIONAL FIGURES")
print("=" * 70)

# ── Figure 9: SDT d' Bar Plot ──
fig, ax = plt.subplots(figsize=(8, 6))
# d' by condition x target type
sns.barplot(
    data=sdt_df, x="target_type", y="d_prime", hue="condition",
    hue_order=["AB", "NB"], order=["EM", "BB"],
    palette=cond_palette, **_err_kw, **bar_kw, ax=ax,
)
ax.set_xlabel("Target Frame Type")
ax.set_ylabel("d' (Discriminability)")
ax.set_title("Signal Detection: d' by Condition and Target Type")
ax.legend(title="Boundary Type")
# Note: Criterion c is not plotted — undefined in 2AFC (always ~0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig9_sdt_dprime.png"))
plt.close()
print("  Saved fig9_sdt_dprime.png")

# ── Figure 10: RT Interaction Plot with Individual Data ──
fig, ax = plt.subplots(figsize=(8, 6))
rt_interaction = (
    subj_means.dropna(subset=["rt"])
    .groupby(["condition", "target_type"])
    .agg(rt_mean=("rt", "mean"), rt_se=("rt", lambda x: x.std() / np.sqrt(len(x))))
    .reset_index()
)
for cond, color, marker in [("AB", AB_COLOR, "s"), ("NB", NB_COLOR, "o")]:
    d = rt_interaction[rt_interaction["condition"] == cond]
    ax.errorbar(
        d["target_type"], d["rt_mean"], yerr=d["rt_se"],
        marker=marker, markersize=10, linewidth=2.5, capsize=5,
        color=color, label=f"{cond} ({'Abrupt' if cond == 'AB' else 'Natural'})",
    )
# Overlay individual data points
for cond, color in [("AB", AB_COLOR), ("NB", NB_COLOR)]:
    d = subj_means[(subj_means["condition"] == cond) & subj_means["rt"].notna()]
    x_jitter = np.where(d["target_type"] == "EM", -0.05, 0.05)
    ax.scatter(
        np.where(d["target_type"] == "EM", 0, 1) + x_jitter + np.random.normal(0, 0.02, len(d)),
        d["rt"], alpha=0.15, s=15, color=color, zorder=1,
    )
ax.set_xlabel("Target Frame Type")
ax.set_ylabel("Mean Response Time (s) ± SE")
ax.set_title("RT Interaction: Boundary Type × Target Type")
ax.legend(title="Boundary Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig10_rt_interaction.png"))
plt.close()
print("  Saved fig10_rt_interaction.png")

# ── Figure 11: Demographic Distributions ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# Age histogram by condition
for cond, color in [("AB", AB_COLOR), ("NB", NB_COLOR)]:
    d = subj_demo[subj_demo["condition"] == cond]
    axes[0].hist(d["age_demo"], bins=12, alpha=0.6, color=color,
                 edgecolor="black", linewidth=0.5, label=cond)
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Count")
axes[0].set_title("Age Distribution by Condition")
axes[0].legend()

# Gender bar chart by condition
gender_ct = subj_demo.groupby(["condition", "gender_demo"]).size().unstack(fill_value=0)
gender_ct.plot(kind="bar", ax=axes[1], color=["#F4A259", "#76B041"], edgecolor="black")
axes[1].set_xlabel("Condition")
axes[1].set_ylabel("Count")
axes[1].set_title("Gender Distribution by Condition")
axes[1].legend(title="Gender")
axes[1].tick_params(axis="x", rotation=0)

# Vision bar chart by condition
vision_ct = subj_demo.groupby(["condition", "vision_demo"]).size().unstack(fill_value=0)
vision_ct.plot(kind="bar", ax=axes[2], color=["#2E86AB", "#E4572E", "#76B041"],
               edgecolor="black")
axes[2].set_xlabel("Condition")
axes[2].set_ylabel("Count")
axes[2].set_title("Vision by Condition")
axes[2].legend(title="Vision", fontsize=8)
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig11_demographics.png"))
plt.close()
print("  Saved fig11_demographics.png")

# ── Figure 12: Mixed-Effects Forest Plot (fixed effects) ──
try:
    me_acc = me_data.dropna(subset=["accuracy"]).copy()
    model_acc = smf.mixedlm(
        "accuracy ~ cond_code * tt_code",
        data=me_acc, groups=me_acc["subject_id"],
    )
    result_acc = model_acc.fit(reml=True)

    # Keep only fixed effects (exclude variance components)
    keep = [p for p in result_acc.params.index if "Var" not in p and "Group" not in p]
    params = result_acc.params[keep]
    ci = result_acc.conf_int().loc[keep]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(params))
    ax.errorbar(
        params.values, y_pos,
        xerr=[params.values - ci.iloc[:, 0].values, ci.iloc[:, 1].values - params.values],
        fmt="o", color="#2E86AB", capsize=5, markersize=8,
    )
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(list(y_pos))
    labels = {"Intercept": "Intercept", "cond_code": "Condition (NB)",
              "tt_code": "Target (EM)", "cond_code:tt_code": "Interaction"}
    ax.set_yticklabels([labels.get(p, p) for p in params.index])
    ax.set_xlabel("Estimate (95% CI)")
    ax.set_title("Mixed-Effects Model: Fixed Effects on Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig12_mixed_effects_forest.png"))
    plt.close()
    print("  Saved fig12_mixed_effects_forest.png")
except Exception as e:
    print(f"  Forest plot error: {e}")

# ── Figure 13: Confidence Calibration Plot ──
# Shows accuracy at each confidence level — are participants well-calibrated?
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: By Condition
ax = axes[0]
for cond, color, marker in [("AB", AB_COLOR, "s"), ("NB", NB_COLOR, "o")]:
    cond_trials = trials[(trials["condition"] == cond) & trials["conf"].notna()].copy()
    conf_levels = sorted(cond_trials["conf"].dropna().unique())
    acc_means, acc_ses, counts = [], [], []
    for cl in conf_levels:
        subset = cond_trials[cond_trials["conf"] == cl]
        acc_means.append(subset["accuracy"].mean())
        acc_ses.append(subset["accuracy"].std() / np.sqrt(len(subset)) if len(subset) > 1 else 0)
        counts.append(len(subset))
    ax.errorbar(
        conf_levels, acc_means, yerr=acc_ses,
        marker=marker, markersize=9, linewidth=2, capsize=4,
        color=color, label=f"{cond} ({'Abrupt' if cond == 'AB' else 'Natural'})",
    )
    # Annotate counts
    for cl, cnt in zip(conf_levels, counts):
        ax.annotate(f"n={cnt}", (cl, 0.48), fontsize=7, ha="center", color=color, alpha=0.7)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Confidence Rating")
ax.set_ylabel("Mean Accuracy")
ax.set_title("Confidence Calibration by Condition")
ax.set_ylim(0.45, 1.0)
ax.legend(title="Boundary Type")

# Panel 2: By Target Type
ax = axes[1]
for tt, color, marker in [("EM", "#76B041", "o"), ("BB", "#F4A259", "s")]:
    tt_trials = trials[(trials["target_type"] == tt) & trials["conf"].notna()].copy()
    conf_levels = sorted(tt_trials["conf"].dropna().unique())
    acc_means, acc_ses = [], []
    for cl in conf_levels:
        subset = tt_trials[tt_trials["conf"] == cl]
        acc_means.append(subset["accuracy"].mean())
        acc_ses.append(subset["accuracy"].std() / np.sqrt(len(subset)) if len(subset) > 1 else 0)
    ax.errorbar(
        conf_levels, acc_means, yerr=acc_ses,
        marker=marker, markersize=9, linewidth=2, capsize=4,
        color=color, label=f"{tt} ({'Event-Model' if tt == 'EM' else 'Boundary-Break'})",
    )
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Confidence Rating")
ax.set_ylabel("Mean Accuracy")
ax.set_title("Confidence Calibration by Target Type")
ax.set_ylim(0.45, 1.0)
ax.legend(title="Target Type")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig13_confidence_calibration.png"))
plt.close()
print("  Saved fig13_confidence_calibration.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE UPDATED OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
subj_means.to_csv(os.path.join(OUTPUT_DIR, "subject_means.csv"), index=False)
trials.to_csv(os.path.join(OUTPUT_DIR, "all_trials_clean.csv"), index=False)
subj_demo.to_csv(os.path.join(OUTPUT_DIR, "demographics_full.csv"), index=False)

print(f"\n{'=' * 70}")
print("ALL ANALYSES COMPLETE")
print(f"{'=' * 70}")
