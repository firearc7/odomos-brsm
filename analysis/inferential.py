"""Mixed ANOVA and non-parametric follow-up tests."""

import pingouin as pg
from scipy import stats

from .config import DV_LABELS
from .utils import pcol, ucol, wcol


def prepare_anova_data(subj_means):
    """Filter to subjects with both EM and BB target types."""
    subj_both = subj_means.groupby("subject_id")["target_type"].nunique()
    valid_subjs = subj_both[subj_both == 2].index
    anova_data = subj_means[subj_means["subject_id"].isin(valid_subjs)].copy()
    print(
        f"\nSubjects with both EM and BB: {len(valid_subjs)} "
        f"(dropped {subj_means['subject_id'].nunique() - len(valid_subjs)})"
    )
    return anova_data


def _run_follow_ups(dv_data, dv, row, pc):
    """Print follow-up tests for significant ANOVA effects."""
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


def _run_nonparametric(dv_data, dv, normality_ok):
    """Run rank-based alternatives when normality is violated."""
    if normality_ok[dv]:
        return

    print(f"\n  [Non-parametric] Normality violated — running rank-based tests:")

    # Between-subjects: Mann-Whitney U on subject-level means
    ab_means = dv_data[dv_data["condition"] == "AB"].groupby("subject_id")[dv].mean()
    nb_means = dv_data[dv_data["condition"] == "NB"].groupby("subject_id")[dv].mean()
    try:
        mwu = pg.mwu(ab_means, nb_means, alternative="two-sided")
        mwu_pc = pcol(mwu)
        U_val = mwu[ucol(mwu)].values[0]
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
            wil_pc = pcol(wil)
            W_val = wil[wcol(wil)].values[0]
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
                mwu_int_pc = pcol(mwu_int)
                U_int = mwu_int[ucol(mwu_int)].values[0]
                p_int = mwu_int[mwu_int_pc].values[0]
                rbc_int = mwu_int["RBC"].values[0]
                print(
                    f"    Interaction (Mann-Whitney on EM-BB diff): U = {U_int:.1f}, "
                    f"p = {p_int:.4f}, rank-biserial r = {rbc_int:.3f}"
                )

                if p_int < 0.05:
                    print("    Non-parametric simple effects:")
                    for tt in ["EM", "BB"]:
                        s = dv_data[dv_data["target_type"] == tt]
                        ab_v = s[s["condition"] == "AB"][dv]
                        nb_v = s[s["condition"] == "NB"][dv]
                        mw_s = pg.mwu(ab_v, nb_v, alternative="two-sided")
                        mw_s_pc = pcol(mw_s)
                        print(
                            f"      {tt}: AB vs NB — U = {mw_s[ucol(mw_s)].values[0]:.1f}, "
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
                            wil_s_pc = pcol(wil_s)
                            print(
                                f"      {cond_lbl}: EM vs BB — W = {wil_s[wcol(wil_s)].values[0]:.1f}, "
                                f"p = {wil_s[wil_s_pc].values[0]:.4f}"
                            )
            except Exception as e:
                print(f"    Interaction test error: {e}")


def run_inferential_stats(subj_means, normality_ok):
    """Run 2x2 mixed ANOVAs and non-parametric alternatives."""
    print("\n" + "=" * 70)
    print("INFERENTIAL STATISTICS")
    print("=" * 70)

    anova_data = prepare_anova_data(subj_means)

    for dv in ["accuracy", "rt", "conf"]:
        print(f"\n{'─' * 60}")
        print(f"DV: {DV_LABELS[dv]}")
        print(f"{'─' * 60}")

        dv_data = anova_data.dropna(subset=[dv]).copy()
        subj_check = dv_data.groupby("subject_id")["target_type"].nunique()
        ok_subjs = subj_check[subj_check == 2].index
        dv_data = dv_data[dv_data["subject_id"].isin(ok_subjs)]

        n_ab_dv = dv_data[dv_data["condition"] == "AB"]["subject_id"].nunique()
        n_nb_dv = dv_data[dv_data["condition"] == "NB"]["subject_id"].nunique()
        print(f"  N: AB = {n_ab_dv}, NB = {n_nb_dv}, Total = {n_ab_dv + n_nb_dv}")

        # ── Parametric: Mixed ANOVA ──
        print("\n  [Parametric] 2x2 Mixed ANOVA:")
        try:
            aov = pg.mixed_anova(
                data=dv_data,
                dv=dv,
                between="condition",
                within="target_type",
                subject="subject_id",
            )
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

            for _, row in aov.iterrows():
                if row[pc] < 0.05:
                    _run_follow_ups(dv_data, dv, row, pc)

        except Exception as e:
            print(f"    ERROR running ANOVA: {e}")

        _run_nonparametric(dv_data, dv, normality_ok)

    return anova_data
