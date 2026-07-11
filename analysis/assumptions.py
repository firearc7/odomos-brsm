"""Normality and homogeneity of variance checks."""

import os

import matplotlib.pyplot as plt
from scipy import stats

from .config import DV_LABELS, OUTPUT_DIR


def run_assumption_checks(subj_means):
    """Run Shapiro-Wilk and Levene tests; save QQ plots."""
    print("\n" + "=" * 70)
    print("ASSUMPTION CHECKS")
    print("=" * 70)

    normality_ok = {}

    for dv in ["accuracy", "rt", "conf"]:
        print(f"\n── {DV_LABELS[dv]} ──")

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
            ax.set_title(f"{cond} x {tt}\n({DV_LABELS[dv]})", fontsize=9)
            ax.get_lines()[0].set_markersize(3)
            ax.get_lines()[0].set_markerfacecolor("steelblue")
    fig.suptitle("QQ Plots for Normality Assessment", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_qq_plots.png"))
    plt.close()
    print("\n  Saved fig5_qq_plots.png")

    return normality_ok
