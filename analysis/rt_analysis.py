"""RT interaction deep-dive analysis (H4)."""

import pingouin as pg
from scipy import stats


def run_rt_analysis(anova_data):
    """Simple effects and Bayesian t-test on RT interaction."""
    print("\n" + "=" * 70)
    print("RT INTERACTION ANALYSIS (H4)")
    print("=" * 70)

    rt_data = anova_data.dropna(subset=["rt"]).copy()
    rt_check = rt_data.groupby("subject_id")["target_type"].nunique()
    rt_ok = rt_check[rt_check == 2].index
    rt_data = rt_data[rt_data["subject_id"].isin(rt_ok)]

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

    for tt in ["EM", "BB"]:
        s = rt_data[rt_data["target_type"] == tt]
        ab_v = s[s["condition"] == "AB"]["rt"]
        nb_v = s[s["condition"] == "NB"]["rt"]
        t_val, p_val = stats.ttest_ind(ab_v, nb_v)
        d_val = pg.compute_effsize(ab_v, nb_v, eftype="cohen")
        print(f"    {tt}: AB (M={ab_v.mean():.3f}) vs NB (M={nb_v.mean():.3f}) — "
              f"t = {t_val:.3f}, p = {p_val:.4f}, d = {d_val:.3f}")

    rt_wide = rt_data.pivot_table(index="subject_id", columns="target_type", values="rt")
    rt_diff = (rt_wide["EM"] - rt_wide["BB"]).dropna()
    cond_map_rt = rt_data.drop_duplicates("subject_id").set_index("subject_id")["condition"]
    ab_rt_diff = rt_diff[rt_diff.index.isin(cond_map_rt[cond_map_rt == "AB"].index)]
    nb_rt_diff = rt_diff[rt_diff.index.isin(cond_map_rt[cond_map_rt == "NB"].index)]

    print(f"\n  RT difference scores (EM - BB):")
    print(f"    AB: M = {ab_rt_diff.mean():.3f}, SD = {ab_rt_diff.std():.3f}")
    print(f"    NB: M = {nb_rt_diff.mean():.3f}, SD = {nb_rt_diff.std():.3f}")

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
