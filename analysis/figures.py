"""Publication-quality figures for the Movie Memory experiment."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf

from .config import (
    AB_COLOR,
    BAR_KW,
    COND_PALETTE,
    ERR_KW,
    NB_COLOR,
    OUTPUT_DIR,
    TT_PALETTE,
)


def plot_core_figures(subj_means, trials):
    """Generate figures 1–8 (core analysis plots)."""
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

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
        palette=COND_PALETTE, **ERR_KW, **BAR_KW, ax=ax,
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
        palette=COND_PALETTE, **ERR_KW, **BAR_KW, ax=ax,
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
        palette=TT_PALETTE, split=True, inner="quartile", alpha=0.7, ax=ax,
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

    # ── Figure 8: Task Paradigm Diagram ──
    fig, ax = plt.subplots(figsize=(9.35, 5.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 94, "Experimental Paradigm", ha="center", va="center", fontsize=18, fontweight="bold", color="#1e293b")
    ax.text(24, 85, "Phase 1: Encoding", ha="center", va="center", fontsize=15, fontweight="bold", color="#0284c7")
    ax.text(76, 85, "Phase 2: Recognition (2AFC)", ha="center", va="center", fontsize=15, fontweight="bold", color="#ea580c")

    def draw_box(x, y, text, border_color, fill_color, fontsize=11, style="round,pad=0.6", ha="center"):
        bbox = dict(boxstyle=style, facecolor=fill_color, edgecolor=border_color, linewidth=1.5)
        ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color="#1e293b", bbox=bbox, zorder=3)

    draw_box(24, 68, "Watch 40 Movie Clips\n(Abrupt vs. Natural Boundaries)", "#3b82f6", "#eff6ff")

    arrow_kw = dict(arrowstyle="-|>,head_width=0.5,head_length=0.6", color="#475569", linewidth=2.5)
    ax.annotate("", xy=(58, 68), xytext=(44, 68), arrowprops=arrow_kw, zorder=2)
    ax.text(51, 73, "Retention Delay", ha="center", va="center", fontsize=11, fontstyle="italic", color="#475569")

    draw_box(76, 68, "Target Frame vs. Lure Frame", "#f59e0b", "#fef3c7")
    ax.annotate("", xy=(76, 56), xytext=(76, 62), arrowprops=arrow_kw, zorder=2)

    draw_box(76, 46, "Target Frame Types:\n1. Event-Model (EM)\n2. Boundary-Break (BB)", "#22c55e", "#dcfce7")
    ax.annotate("", xy=(76, 33), xytext=(76, 39), arrowprops=arrow_kw, zorder=2)

    draw_box(76, 26, "Confidence Rating (1-5)", "#a855f7", "#f3e8ff")

    loop_kw = dict(arrowstyle="-|>,head_width=0.4,head_length=0.6", color="#64748b", linewidth=2, linestyle="--")
    ax.plot([86, 94, 94], [26, 26, 68], color="#64748b", linewidth=2, linestyle="--", zorder=1)
    ax.annotate("", xy=(87, 68), xytext=(94, 68), arrowprops=loop_kw, zorder=2)
    ax.text(97, 47, "x 40 trials", ha="center", va="center", fontsize=11, fontstyle="italic", color="#64748b", rotation=270)

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


def plot_extended_figures(subj_means, trials, sdt_df, subj_demo, me_data):
    """Generate figures 9–13 (extended analysis plots)."""
    print("\n" + "=" * 70)
    print("ADDITIONAL FIGURES")
    print("=" * 70)

    # ── Figure 9: SDT d' Bar Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=sdt_df, x="target_type", y="d_prime", hue="condition",
        hue_order=["AB", "NB"], order=["EM", "BB"],
        palette=COND_PALETTE, **ERR_KW, **BAR_KW, ax=ax,
    )
    ax.set_xlabel("Target Frame Type")
    ax.set_ylabel("d' (Discriminability)")
    ax.set_title("Signal Detection: d' by Condition and Target Type")
    ax.legend(title="Boundary Type")
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
    for cond, color in [("AB", AB_COLOR), ("NB", NB_COLOR)]:
        d = subj_demo[subj_demo["condition"] == cond]
        axes[0].hist(d["age_demo"], bins=12, alpha=0.6, color=color,
                     edgecolor="black", linewidth=0.5, label=cond)
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Age Distribution by Condition")
    axes[0].legend()

    gender_ct = subj_demo.groupby(["condition", "gender_demo"]).size().unstack(fill_value=0)
    gender_ct.plot(kind="bar", ax=axes[1], color=["#F4A259", "#76B041"], edgecolor="black")
    axes[1].set_xlabel("Condition")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Gender Distribution by Condition")
    axes[1].legend(title="Gender")
    axes[1].tick_params(axis="x", rotation=0)

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
        for cl, cnt in zip(conf_levels, counts):
            ax.annotate(f"n={cnt}", (cl, 0.48), fontsize=7, ha="center", color=color, alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Confidence Rating")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Confidence Calibration by Condition")
    ax.set_ylim(0.45, 1.0)
    ax.legend(title="Boundary Type")

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
