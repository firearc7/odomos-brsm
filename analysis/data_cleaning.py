"""Trial-level cleaning and per-subject aggregation."""

import numpy as np
import pandas as pd


def clean_trials(trials, vigilance_df):
    """Drop invalid trials, flag low performers, and remove RT outliers."""
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

    return trials


def compute_subject_means(trials):
    """Aggregate trial-level data to per-subject means by condition and target type."""
    return (
        trials.groupby(["subject_id", "condition", "target_type"])
        .agg(
            accuracy=("accuracy", "mean"),
            rt=("rt", "mean"),
            conf=("conf", "mean"),
            n_trials=("accuracy", "count"),
        )
        .reset_index()
    )
