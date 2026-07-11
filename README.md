# Movie Memory Experiment — Team Odomos

Statistical analysis pipeline for a BRSM study investigating how **boundary type** (Abrupt Cut vs. Natural Cut) and **target frame type** (Event-Model vs. Boundary-Break) affect recognition memory for movie clips.

**Authors:** Archit Choudhary (2023114002), Bhavya Ahuja (2023111035), Hrishiraj Mitra (2023111037)

## Overview

Participants viewed 40 short movie clips under one of two encoding conditions:

| Condition | Description |
|-----------|-------------|
| **AB** (Abrupt Boundary) | Clips interrupted 1–5 s before a natural event boundary |
| **NB** (Natural Boundary) | Clips played with original, uninterrupted transitions |

After encoding, all participants completed a **two-alternative forced-choice (2AFC)** recognition test with confidence ratings. Target frames were either **Event-Model (EM)** or **Boundary-Break (BB)** frames.

### Design

2 (Boundary Type: AB vs. NB; between-subjects) × 2 (Target Type: EM vs. BB; within-subjects)

### Dependent Variables

- Recognition accuracy
- Response time (RT)
- Confidence rating (1–5)

### Hypotheses

| ID | Prediction |
|----|------------|
| H1 | NB > AB in recognition accuracy |
| H2 | BB > EM in recognition accuracy (boundary advantage) |
| H3 | Condition × Target Type interaction on accuracy |
| H4 | Condition × Target Type interaction on RT |
| H5 | NB shows higher d′ than AB |
| H6 | Demographics (age, gender, vision) moderate accuracy |

## Project Structure

```
odomos-brsm/
├── analysis.py              # Entry point — run the full pipeline
├── analysis/                # Modular analysis package
│   ├── config.py            # Paths, plot styling, constants
│   ├── utils.py             # Parsing helpers (RT, filenames, pingouin columns)
│   ├── data_loading.py      # Load PsychoPy recognition-stage CSVs
│   ├── data_cleaning.py     # Trial cleaning and subject-level aggregation
│   ├── descriptive.py       # Descriptive statistics
│   ├── assumptions.py       # Normality checks and QQ plots
│   ├── inferential.py       # Mixed ANOVA and non-parametric tests
│   ├── correlations.py      # DV correlations
│   ├── demographics.py      # External demographic merge and moderation (H6)
│   ├── sdt.py               # Signal Detection Theory (H5)
│   ├── mixed_effects.py     # Linear mixed-effects models
│   ├── rt_analysis.py       # RT interaction deep-dive (H4)
│   ├── figures.py           # All publication-quality figures
│   └── pipeline.py          # Orchestrates the full analysis workflow
├── data/                    # Subject CSV files (gitignored)
├── demographic_data.csv     # External demographic survey (gitignored)
├── output/                  # Generated CSVs and figures
├── report.md                # Full BRSM report (Pandoc/LaTeX source)
├── presentation.md          # Conference presentation source
├── poster.tex               # Poster LaTeX source
└── requirements.txt         # Python dependencies
```

## Setup

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place raw PsychoPy output files in `data/` (or the project root). Expected filename pattern:

```
sub{N}_{AB|NB}_recognitionstage_*.csv
```

Also place `demographic_data.csv` in the project root. Both CSV types are gitignored to protect participant data.

## Running the Analysis

```bash
python analysis.py
```

This executes the full pipeline:

1. Load and parse subject CSVs
2. Clean trials (drop missing accuracy, remove RT outliers)
3. Compute descriptive statistics
4. Check normality assumptions (Shapiro-Wilk, Levene, QQ plots)
5. Run 2×2 mixed ANOVAs with follow-up tests
6. Run non-parametric alternatives where normality is violated
7. Compute DV correlations
8. Generate core figures (fig1–fig8)
9. Merge external demographic data
10. Signal Detection Theory analysis (d′)
11. Mixed-effects models with crossed random effects
12. RT interaction analysis with Bayesian t-test
13. Demographic moderation (ANCOVA)
14. Generate extended figures (fig9–fig13)
15. Save all output CSVs

## Output

All results are written to `output/`:

| File | Description |
|------|-------------|
| `descriptive_statistics.csv` | Means and SDs by condition × target type |
| `subject_means.csv` | Per-subject aggregated data |
| `all_trials_clean.csv` | Cleaned trial-level data |
| `demographics.csv` | Demographics extracted from PsychoPy files |
| `demographics_full.csv` | Merged external demographic data |
| `sdt_results.csv` | Signal detection (d′) per subject |
| `fig1_accuracy_interaction.png` | Accuracy interaction plot |
| `fig2_rt_barplot.png` | RT bar plot |
| `fig3_confidence_barplot.png` | Confidence bar plot |
| `fig4_accuracy_violin.png` | Accuracy distribution |
| `fig5_qq_plots.png` | Normality QQ plots |
| `fig6_accuracy_histogram.png` | Subject-level accuracy histogram |
| `fig7_confidence_interaction.png` | Confidence interaction plot |
| `fig8_task_paradigm.png` | Experimental paradigm diagram |
| `fig9_sdt_dprime.png` | d′ bar plot |
| `fig10_rt_interaction.png` | RT interaction plot |
| `fig11_demographics.png` | Demographic distributions |
| `fig12_mixed_effects_forest.png` | Mixed-effects forest plot |
| `fig13_confidence_calibration.png` | Confidence calibration |

## Documentation

- [`report.md`](report.md) — Full written report with methods, results, and discussion
- [`presentation.md`](presentation.md) — Slide deck source
- [`poster.tex`](poster.tex) — Conference poster

## References

- Radvansky, G. A., & Zacks, R. T. (2017). Event boundaries in memory and cognition. *Current Opinion in Behavioral Sciences*, 17, 133–140.
- Zacks, J. M., Speer, N. K., Swallow, K. M., Braver, T. S., & Reynolds, J. R. (2007). Event perception: A mind-brain perspective. *Psychological Bulletin*, 133(2), 273–293.
- Swallow, K. M., Zacks, J. M., & Abrams, R. A. (2009). Event boundaries in perception affect memory encoding and updating. *Journal of Experimental Psychology: General*, 138(2), 236–257.

## License

Academic research project — IIT Delhi BRSM course submission.
