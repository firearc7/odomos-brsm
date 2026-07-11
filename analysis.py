"""
Movie Memory Experiment — Full Data Analysis
=============================================
Team Odomos
Archit Choudhary (2023114002), Bhavya Ahuja (2023111035), Hrishiraj Mitra (2023111037)

Design: 2 (Boundary Type: AB vs NB; between) x 2 (Target Type: EM vs BB; within)
DVs: Recognition Accuracy, Response Time, Confidence Rating

Hypotheses:
  H1: NB > AB in recognition accuracy (boundary type main effect)
  H2: BB > EM in recognition accuracy (boundary advantage; Radvansky & Zacks, 2017)
  H3: Condition x Target Type interaction on accuracy
  H4: Condition x Target Type interaction on RT (AB slows for BB)
  H5: NB shows higher d' (signal detection) than AB
  H6: Demographics (age, gender, vision) moderate recognition accuracy

Analyses: descriptive statistics, normality checks, parametric/non-parametric
tests, Signal Detection Theory, mixed-effects models with crossed random
effects, demographic moderation, and publication-quality figures.
"""

from analysis.pipeline import run_analysis

if __name__ == "__main__":
    run_analysis()
