"""Signal Detection Theory analysis (H5)."""

import os

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

from .config import OUTPUT_DIR
from .utils import pcol, ucol


def run_sdt_analysis(trials):
    """Compute d' and compare discriminability across conditions."""
    print("\n" + "=" * 70)
    print("SIGNAL DETECTION THEORY (H5)")
    print("=" * 70)

    sdt_list = []
    for (sid, cond), grp in trials.groupby(["subject_id", "condition"]):
        for tt in ["EM", "BB"]:
            tt_trials = grp[grp["target_type"] == tt]
            n = len(tt_trials)
            if n == 0:
                continue
            n_correct = tt_trials["accuracy"].sum()
            hr = (n_correct + 0.5) / (n + 1)
            far = 1 - hr
            d_prime = stats.norm.ppf(hr) - stats.norm.ppf(far)
            sdt_list.append({
                "subject_id": sid, "condition": cond, "target_type": tt,
                "hit_rate": hr, "false_alarm_rate": far,
                "d_prime": d_prime, "n_trials": n,
            })

    sdt_df = pd.DataFrame(sdt_list)

    print("\n  d' by Condition and Target Type (M ± SD):")
    sdt_desc = sdt_df.groupby(["condition", "target_type"]).agg(
        d_M=("d_prime", "mean"), d_SD=("d_prime", "std"),
        N=("subject_id", "nunique"),
    ).reset_index()
    for _, r in sdt_desc.iterrows():
        print(f"    {r['condition']} x {r['target_type']}: "
              f"d' = {r['d_M']:.3f} ± {r['d_SD']:.3f}")

    sdt_subj = sdt_df.groupby(["subject_id", "condition"])["d_prime"].mean().reset_index()
    ab_dp = sdt_subj[sdt_subj["condition"] == "AB"]["d_prime"]
    nb_dp = sdt_subj[sdt_subj["condition"] == "NB"]["d_prime"]

    print("\n  H5: d' comparison (NB > AB?)")
    _, p_sw_ab = stats.shapiro(ab_dp)
    _, p_sw_nb = stats.shapiro(nb_dp)
    print(f"    Shapiro-Wilk: AB p = {p_sw_ab:.4f}, NB p = {p_sw_nb:.4f}")

    if p_sw_ab >= 0.05 and p_sw_nb >= 0.05:
        t_val, p_val = stats.ttest_ind(nb_dp, ab_dp)
        d_val = pg.compute_effsize(nb_dp, ab_dp, eftype="cohen")
        print(f"    Independent t-test: t = {t_val:.3f}, p = {p_val:.4f}, d = {d_val:.3f}")
    else:
        mwu = pg.mwu(nb_dp, ab_dp, alternative="two-sided")
        pc = pcol(mwu)
        print(f"    Mann-Whitney U = {mwu[ucol(mwu)].values[0]:.1f}, "
              f"p = {mwu[pc].values[0]:.4f}, RBC = {mwu['RBC'].values[0]:.3f}")

    t_val_dp, p_val_dp = stats.ttest_ind(nb_dp, ab_dp)
    d_val_dp = pg.compute_effsize(nb_dp, ab_dp, eftype="cohen")
    print(f"    t = {t_val_dp:.3f}, p = {p_val_dp:.4f}, d = {d_val_dp:.3f}")
    print(f"    AB d' M = {ab_dp.mean():.3f}, NB d' M = {nb_dp.mean():.3f}")

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

    return sdt_df
