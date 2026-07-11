"""External demographic data integration and moderation analysis (H6)."""

import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import stats

from .config import DEMO_CSV


def merge_demographics(trials, subj_means):
    """Load external demographic CSV and merge into trial/subject data."""
    print("\n" + "=" * 70)
    print("DEMOGRAPHIC DATA INTEGRATION")
    print("=" * 70)

    demo_ext = pd.read_csv(DEMO_CSV)
    demo_ext.columns = demo_ext.columns.str.strip()
    demo_ext = demo_ext.rename(columns={
        "Sub ID": "sub_id_raw", "Age": "age_demo",
        "Gender": "gender_demo", "Handedness": "hand_demo", "Vision": "vision_demo",
    })
    demo_ext["subject_id"] = (
        demo_ext["sub_id_raw"].astype(str).str.strip().str.lower()
        .str.replace(r"_[an]b$", "", regex=True)
    )
    demo_ext = demo_ext.drop_duplicates(subset="subject_id", keep="first")

    age_median = demo_ext["age_demo"].median()
    demo_ext["age_demo"] = demo_ext["age_demo"].fillna(age_median)
    for col in ["gender_demo", "hand_demo", "vision_demo"]:
        mode_val = demo_ext[col].mode().iloc[0] if len(demo_ext[col].mode()) > 0 else "Unknown"
        demo_ext[col] = demo_ext[col].fillna(mode_val)

    print(f"  Demographic CSV: {len(demo_ext)} entries")
    print(f"  Age imputed with median ({age_median:.0f}) for missing values")
    print(f"  Categorical vars imputed with mode for missing values")

    trials = trials.merge(
        demo_ext[["subject_id", "age_demo", "gender_demo", "hand_demo", "vision_demo"]],
        on="subject_id", how="left",
    )
    trials["age_demo"] = trials["age_demo"].fillna(age_median)
    for col in ["gender_demo", "hand_demo", "vision_demo"]:
        mode_val = trials[col].mode().iloc[0] if len(trials[col].mode()) > 0 else "Unknown"
        trials[col] = trials[col].fillna(mode_val)

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

    return trials, subj_means


def run_demographic_moderation(trials):
    """Test whether demographics moderate recognition accuracy (H6)."""
    print("\n" + "=" * 70)
    print("DEMOGRAPHIC MODERATION (H6)")
    print("=" * 70)

    subj_demo = (
        trials.groupby(["subject_id", "condition", "age_demo", "gender_demo",
                         "hand_demo", "vision_demo"])
        .agg(accuracy=("accuracy", "mean"), rt=("rt", "mean"), conf=("conf", "mean"))
        .reset_index()
    )

    print("\n  Age correlations:")
    for dv in ["accuracy", "rt", "conf"]:
        valid = subj_demo[["age_demo", dv]].dropna()
        if len(valid) >= 3:
            rho, p_rho = stats.spearmanr(valid["age_demo"], valid[dv])
            print(f"    Age vs {dv}: rho = {rho:.3f}, p = {p_rho:.4f}")

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
        try:
            ols_model = smf.ols(
                "accuracy ~ C(condition) + age_demo + gender_code + vision_code",
                data=subj_demo,
            ).fit()
            print(ols_model.summary2().tables[1].to_string())
        except Exception as e2:
            print(f"    OLS error: {e2}")

    return subj_demo
