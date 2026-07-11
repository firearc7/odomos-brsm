"""Main analysis pipeline — orchestrates all modules."""

import os

from .assumptions import run_assumption_checks
from .config import OUTPUT_DIR
from .correlations import run_correlations
from .data_cleaning import clean_trials, compute_subject_means
from .data_loading import load_subject_files
from .demographics import merge_demographics, run_demographic_moderation
from .descriptive import run_descriptive_stats
from .figures import plot_core_figures, plot_extended_figures
from .inferential import run_inferential_stats
from .mixed_effects import prepare_mixed_effects_data, run_mixed_effects_models
from .rt_analysis import run_rt_analysis
from .sdt import run_sdt_analysis


def run_analysis():
    """Execute the full Movie Memory experiment analysis pipeline."""
    print("=" * 70)
    print("MOVIE MEMORY EXPERIMENT — FULL ANALYSIS")
    print("Team Odomos")
    print("=" * 70)

    # 1. Load data
    trials, demographics, vigilance_df, _ = load_subject_files()

    # 2–3. Clean and aggregate
    trials = clean_trials(trials, vigilance_df)
    subj_means = compute_subject_means(trials)

    # 4. Descriptive statistics
    run_descriptive_stats(subj_means)

    # 5. Assumption checks
    normality_ok = run_assumption_checks(subj_means)

    # 6. Inferential statistics
    anova_data = run_inferential_stats(subj_means, normality_ok)

    # 7. Correlations
    run_correlations(trials)

    # 8. Core figures
    plot_core_figures(subj_means, trials)

    # 9. Save initial outputs
    subj_means.to_csv(os.path.join(OUTPUT_DIR, "subject_means.csv"), index=False)
    trials.to_csv(os.path.join(OUTPUT_DIR, "all_trials_clean.csv"), index=False)
    demographics.to_csv(os.path.join(OUTPUT_DIR, "demographics.csv"), index=False)
    print("\nOriginal outputs saved. Running extended analyses...")

    # 10. Demographic integration
    trials, subj_means = merge_demographics(trials, subj_means)

    # 11. Signal Detection Theory
    sdt_df = run_sdt_analysis(trials)

    # 12. Mixed-effects models
    me_data = prepare_mixed_effects_data(trials)
    run_mixed_effects_models(me_data)

    # 13. RT interaction deep-dive
    run_rt_analysis(anova_data)

    # 14. Demographic moderation
    subj_demo = run_demographic_moderation(trials)

    # 15. Extended figures
    plot_extended_figures(subj_means, trials, sdt_df, subj_demo, me_data)

    # 16. Save final outputs
    subj_means.to_csv(os.path.join(OUTPUT_DIR, "subject_means.csv"), index=False)
    trials.to_csv(os.path.join(OUTPUT_DIR, "all_trials_clean.csv"), index=False)
    subj_demo.to_csv(os.path.join(OUTPUT_DIR, "demographics_full.csv"), index=False)

    print(f"\n{'=' * 70}")
    print("ALL ANALYSES COMPLETE")
    print(f"{'=' * 70}")
