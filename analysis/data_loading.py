"""Load and parse PsychoPy recognition-stage CSV files."""

import glob
import os

import pandas as pd

from .config import BASE_DIR, DATA_DIR
from .utils import (
    extract_condition,
    extract_subject_id,
    extract_target_type,
    parse_rt,
)


def load_subject_files():
    """Discover and parse all subject CSV files.

    Returns
    -------
    trials : pd.DataFrame
    demographics : pd.DataFrame
    vigilance_df : pd.DataFrame
    skipped : list[tuple[str, str]]
    """
    csv_files = (
        glob.glob(os.path.join(DATA_DIR, "sub*_recognitionstage_*.csv"))
        + glob.glob(os.path.join(DATA_DIR, "Sub*_recognitionstage_*.csv"))
        + glob.glob(os.path.join(DATA_DIR, "sub*_AB *_recognitionstage_*.csv"))
        + glob.glob(os.path.join(DATA_DIR, "sub*_NB *_recognitionstage_*.csv"))
        + glob.glob(os.path.join(BASE_DIR, "sub*_recognitionstage_*.csv"))
        + glob.glob(os.path.join(BASE_DIR, "Sub*_recognitionstage_*.csv"))
    )
    csv_files = list(set(csv_files))

    print(f"\nFound {len(csv_files)} subject CSV files.")

    all_trials = []
    demographics_list = []
    vigilance_list = []
    skipped = []

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            skipped.append((fpath, str(e)))
            continue

        condition = extract_condition(fpath)
        subject_id = extract_subject_id(fpath)
        if condition is None or subject_id is None:
            skipped.append((fpath, "Could not determine condition or subject ID"))
            continue

        # Demographics
        demo_row = {"subject_id": subject_id, "condition": condition}
        if len(df) > 1:
            for col in ["age", "gender", "handedness", "vision",
                         "caffeine_2h", "alcohol_smoke_12h"]:
                if col in df.columns:
                    demo_row[col] = df[col].iloc[0]
        demographics_list.append(demo_row)

        # Recognition trials: rows where movie_id is not empty
        if "movie_id" not in df.columns:
            skipped.append((fpath, "No movie_id column"))
            continue

        recog = df[df["movie_id"].notna() & (df["movie_id"] != "")].copy()
        if len(recog) == 0:
            skipped.append((fpath, "No recognition trials"))
            continue

        recog["movie_id"] = pd.to_numeric(recog["movie_id"], errors="coerce")
        recog = recog[recog["movie_id"].notna()].copy()

        # Accuracy
        if "resp.corr" in recog.columns:
            recog["accuracy"] = pd.to_numeric(recog["resp.corr"], errors="coerce")
        if "recogloop.resp.corr" in recog.columns:
            alt_acc = pd.to_numeric(recog["recogloop.resp.corr"], errors="coerce")
            if "accuracy" in recog.columns:
                recog["accuracy"] = recog["accuracy"].fillna(alt_acc)
            else:
                recog["accuracy"] = alt_acc

        # RT
        for rt_col in ["resp.rt", "recogloop.resp.rt"]:
            if rt_col in recog.columns:
                parsed = recog[rt_col].apply(parse_rt)
                if "rt" not in recog.columns:
                    recog["rt"] = parsed
                else:
                    recog["rt"] = recog["rt"].fillna(parsed)

        # Confidence
        for conf_col in ["conf_radio.response", "recogloop.conf_radio.response", "confidence"]:
            if conf_col in recog.columns:
                parsed = pd.to_numeric(recog[conf_col], errors="coerce")
                if "conf" not in recog.columns:
                    recog["conf"] = parsed
                else:
                    recog["conf"] = recog["conf"].fillna(parsed)

        # Target type from target_img
        if "target_img" in recog.columns:
            recog["target_type"] = recog["target_img"].apply(extract_target_type)
        else:
            skipped.append((fpath, "No target_img column"))
            continue

        valid = recog[recog["target_type"].notna()].copy()
        if len(valid) == 0:
            skipped.append((fpath, "No valid target types"))
            continue

        valid["subject_id"] = subject_id
        valid["condition"] = condition
        all_trials.append(
            valid[["subject_id", "condition", "movie_id",
                   "target_type", "accuracy", "rt", "conf"]].copy()
        )

        # Vigilance (AB only)
        if condition == "AB" and "vigilance_correct" in df.columns:
            movie_rows = df[df["path"].notna() & (df["path"] != "")].copy()
            if "is_repeat" in movie_rows.columns:
                repeats = movie_rows[
                    pd.to_numeric(movie_rows["is_repeat"], errors="coerce") == 1
                ]
                if len(repeats) > 0:
                    vig_correct = pd.to_numeric(
                        repeats["vigilance_correct"], errors="coerce"
                    )
                    vigilance_list.append({
                        "subject_id": subject_id,
                        "condition": condition,
                        "vigilance_hit_rate": vig_correct.mean(),
                        "n_repeats": len(repeats),
                    })

    if skipped:
        print(f"\nSkipped {len(skipped)} files:")
        for fp, reason in skipped[:10]:
            print(f"  {os.path.basename(fp)}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    trials = pd.concat(all_trials, ignore_index=True)
    demographics = pd.DataFrame(demographics_list)
    vigilance_df = pd.DataFrame(vigilance_list) if vigilance_list else pd.DataFrame()

    n_subj = trials["subject_id"].nunique()
    n_ab = trials[trials["condition"] == "AB"]["subject_id"].nunique()
    n_nb = trials[trials["condition"] == "NB"]["subject_id"].nunique()
    print(f"\nTotal trials: {len(trials)}")
    print(f"Subjects: {n_subj} (AB = {n_ab}, NB = {n_nb})")
    print(f"Trials per subject: ~{len(trials) / n_subj:.0f}")

    return trials, demographics, vigilance_df, skipped
