"""Parsing helpers and pingouin column-name utilities."""

import ast
import os
import re

import numpy as np
import pandas as pd


def parse_rt(val):
    """Parse RT from PsychoPy format — float, string list, or NaN."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s in ("None", "", "[]"):
        return np.nan
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return float(parsed[0]) if len(parsed) > 0 else np.nan
        return float(parsed)
    except Exception:
        try:
            return float(s)
        except ValueError:
            return np.nan


def extract_target_type(target_img):
    """Extract EM or BB from target_img filename."""
    if pd.isna(target_img):
        return np.nan
    s = str(target_img)
    if "_EM_T" in s:
        return "EM"
    elif "_BB_T" in s:
        return "BB"
    return np.nan


def extract_condition(filename):
    """Extract AB or NB from filename (handles spaces)."""
    fn = os.path.basename(filename).upper().replace(" ", "")
    if "_AB_" in fn:
        return "AB"
    elif "_NB_" in fn:
        return "NB"
    return None


def extract_subject_id(filename):
    """Extract subject ID from filename."""
    fn = os.path.basename(filename)
    m = re.match(r"(sub\d+)", fn, re.IGNORECASE)
    return m.group(1).lower() if m else None


def pcol(df):
    """Return the p-value column name in a pingouin result DataFrame."""
    for c in ["p_val", "p-val", "p_unc", "p-unc"]:
        if c in df.columns:
            return c
    return "p_val"


def ucol(df):
    """Return the U-value column name."""
    for c in ["U_val", "U-val"]:
        if c in df.columns:
            return c
    return "U_val"


def wcol(df):
    """Return the W/T-value column name."""
    for c in ["W_val", "W-val", "T_val", "T-val"]:
        if c in df.columns:
            return c
    return "W_val"
