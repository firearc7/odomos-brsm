"""Descriptive statistics and summary tables."""

import os

import pandas as pd

from .config import OUTPUT_DIR


def run_descriptive_stats(subj_means):
    """Print and save descriptive statistics by condition and target type."""
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

    return desc
