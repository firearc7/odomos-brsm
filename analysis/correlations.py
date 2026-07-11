"""Correlations between dependent variables."""

from scipy import stats


def run_correlations(trials):
    """Compute Spearman correlations between DVs at the subject level."""
    print("\n" + "=" * 70)
    print("CORRELATIONS BETWEEN DVs (per-subject level)")
    print("=" * 70)

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

    return subj_overall
