"""Linear mixed-effects models with crossed random effects."""

import numpy as np
import statsmodels.formula.api as smf


def prepare_mixed_effects_data(trials):
    """Code predictors for mixed-effects models."""
    me_data = trials.dropna(subset=["accuracy"]).copy()
    me_data["cond_code"] = (me_data["condition"] == "NB").astype(float)
    me_data["tt_code"] = (me_data["target_type"] == "EM").astype(float)
    me_data["cond_x_tt"] = me_data["cond_code"] * me_data["tt_code"]
    me_data["movie_id"] = me_data["movie_id"].astype(int).astype(str)
    return me_data


def run_mixed_effects_models(me_data):
    """Fit mixed models for accuracy, RT, and confidence."""
    print("\n" + "=" * 70)
    print("MIXED-EFFECTS MODELS (Crossed Random Effects)")
    print("=" * 70)

    for dv, dv_label in [("accuracy", "Accuracy"), ("rt", "RT"), ("conf", "Confidence")]:
        print(f"\n── Mixed Model: {dv_label} ──")
        dv_trials = me_data.dropna(subset=[dv]).copy()

        try:
            vc = {"movie_id": "0 + C(movie_id)"}
            model = smf.mixedlm(
                f"{dv} ~ cond_code * tt_code",
                data=dv_trials, groups=dv_trials["subject_id"],
                vc_formula=vc,
            )
            result = model.fit(reml=True, method="lbfgs")
            print(result.summary().tables[1].to_string())

            for param in ["cond_code", "tt_code", "cond_code:tt_code"]:
                if param in result.params.index:
                    coef = result.params[param]
                    se = result.bse[param]
                    z = result.tvalues[param]
                    p = result.pvalues[param]
                    print(f"    {param}: b = {coef:.4f}, SE = {se:.4f}, z = {z:.3f}, p = {p:.4f}")

            try:
                cre = result.cov_re
                if cre is not None and getattr(cre, "size", 0) > 0:
                    print(f"    Subject RE variance: {float(np.asarray(cre).ravel()[0]):.4f}")
            except Exception:
                print("    Subject RE variance: (not extracted)")

        except Exception as e:
            print(f"    Model failed: {e}")
            try:
                model_simple = smf.mixedlm(
                    f"{dv} ~ cond_code * tt_code",
                    data=dv_trials, groups=dv_trials["subject_id"],
                )
                result_simple = model_simple.fit(reml=True)
                print("    (Fallback: subject-only random intercept)")
                print(result_simple.summary().tables[1].to_string())
            except Exception as e2:
                print(f"    Fallback also failed: {e2}")
