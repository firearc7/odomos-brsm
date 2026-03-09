---
title: "Event Boundaries and Movie Memory"
subtitle: "BRSM Report 1: Team Odomos"
author:
  - Archit Choudhary (2023114002)
  - Bhavya Ahuja (2023111035)
  - Hrishiraj Mitra (2023111037)
date: "March 10, 2026"
institute: "IIIT Hyderabad"
theme: "Madrid"
colortheme: "dolphin"
fonttheme: "structurebold"
fontsize: 11pt
aspectratio: 169
section-titles: false
header-includes:
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \setbeamertemplate{navigation symbols}{}
  - \setbeamertemplate{footline}{\hfill\insertframenumber/\inserttotalframenumber\hspace{2mm}\vspace{2mm}}
---

# Introduction (Archit)

## Background: Event Segmentation Theory

- Our brains **segment** continuous experience into discrete *events*
- Transitions = **event boundaries** (Zacks et al., 2007)
- Boundaries trigger an update of the internal *event model*
- Memory at event boundaries is encoded **differently** (Swallow et al., 2009)

\vspace{0.2cm}

Two types of boundaries:

- **Natural boundaries (NB):** Smooth, organic transitions
- **Abrupt boundaries (AB):** Sudden, artificially imposed transitions (hard cuts)

::: notes
Hello everyone. I am Archit, and I will introduce our study. Event Segmentation Theory, proposed by Zacks and colleagues in 2007, tells us that our brains automatically segment continuous experience into discrete events. The transitions between events, called event boundaries, trigger an update of our internal event model. This is a real-time mental representation of the current situation. This updating has consequences for memory: content near boundaries is encoded differently from mid-event content. There are two types of event boundaries. Natural boundaries are smooth, organic transitions. Abrupt boundaries are sudden, artificially imposed transitions like hard cuts in a film.
:::

## Task Paradigm

![Experimental paradigm: encoding phase (AB vs. NB movie clips), followed by 2AFC recognition test with confidence ratings.](output/fig8_task_paradigm.png){width=85%}

::: notes
This figure shows our experimental paradigm. In the encoding phase, participants watched 40 short movie clips. One group saw clips with abrupt hard cuts at event boundaries, while the other group saw clips with smooth, natural transitions. After a delay, all participants completed a two-alternative forced-choice recognition test with 40 trials. On each trial, they saw a target frame and a perceptually similar lure, and had to identify which frame they had actually seen. They then rated their confidence on a 1 to 5 scale. Target frames were either Event-Model frames, consistent with the ongoing event, or Boundary-Break frames, drawn from near an event boundary. We measured three dependent variables: accuracy, response time, and confidence.
:::

## Design and Participants

- **2 $\times$ 2 mixed design:**
  - Boundary Type (AB vs. NB) -- **between-subjects**
  - Target Frame Type (EM vs. BB) -- **within-subjects**
- **170 participants** (81 AB, 89 NB); 1 excluded for missing data
- 40 trials per participant (20 EM, 20 BB) = **6,800 trials total**
- **DVs:** Recognition accuracy, Response time, Confidence (1--5)

::: notes
We used a two-by-two mixed design. Boundary type, abrupt versus natural, was manipulated between subjects. Target frame type, Event-Model versus Boundary-Break, was manipulated within subjects. Of 171 participants tested, one was excluded for missing recognition data, leaving 170 participants: 81 in the Abrupt Boundary group and 89 in the Natural Boundary group. Each participant completed 40 recognition trials, giving us 6,800 trials in total. Our three dependent variables were recognition accuracy, response time, and confidence on a 1 to 5 scale.
:::

## Hypotheses

**H1 -- Boundary Type main effect:**

- NB group will show better recognition performance than AB group

**H2 -- Target Type main effect:**

- EM targets will be recognized more accurately and confidently than BB targets

**H3 -- Interaction:**

- The EM--BB gap may be larger in the AB group (abrupt boundaries differentially impair boundary-adjacent content)

\vspace{0.2cm}
\small
Each hypothesis tested for all three DVs: Accuracy, RT, Confidence.

::: notes
We had three hypotheses, each tested across all three dependent variables. Hypothesis 1: the Natural Boundary group will show better recognition performance than the Abrupt Boundary group, because natural event segmentation supports more coherent encoding. Hypothesis 2: Event-Model targets will be recognized more accurately and with greater confidence than Boundary-Break targets, because EM frames are consistent with the maintained event representation. Hypothesis 3: the advantage of EM over BB targets may be larger in the Abrupt Boundary group, because abrupt boundaries may differentially impair encoding of boundary-adjacent content. Now I will hand it over to Bhavya for the methods and accuracy results.
:::

# Methods and Accuracy (Bhavya)

## Data Processing and Analytical Pipeline

- **171 PsychoPy CSV files** parsed in Python 3
  - 1 excluded (sub42); Final: **170 participants, 6,800 trials**
- Per trial: accuracy (0/1), RT (s), confidence (1--5)
- RT outliers removed: < 0.2 s or > 60 s (1 trial)

\vspace{0.2cm}

**For each DV, the analysis pipeline was:**

1. Descriptive statistics (means, SDs)
2. **Normality check** (Shapiro-Wilk per cell, Levene's test)
3. 2 $\times$ 2 mixed ANOVA ($\eta_p^2$; follow-ups with Cohen's *d*)
4. **If normality violated** $\rightarrow$ non-parametric robustness checks
   - Mann-Whitney U (between), Wilcoxon (within), Mann-Whitney on diffs (interaction)

::: notes
Hello, I am Bhavya. Let me describe our methods and analysis pipeline. We parsed 171 PsychoPy CSV files. One participant was excluded, leaving 170 participants and 6,800 trials. From each trial, we extracted accuracy, response time, and confidence. One RT trial was removed as an outlier. Critically, for each dependent variable, we followed a systematic four-step pipeline. Step one: compute descriptive statistics. Step two: check normality using Shapiro-Wilk tests on each cell of the design, plus Levene's test for variance homogeneity. Step three: run a two-by-two mixed ANOVA with partial eta-squared and follow up significant effects. Step four: if normality was violated, run non-parametric robustness checks. Mann-Whitney U for between-subjects effects, Wilcoxon signed-rank for within-subjects effects, and Mann-Whitney on difference scores for the interaction.
:::

## H1, H2, H3: Accuracy -- Descriptive Statistics

| Condition | Target | Accuracy (*M* $\pm$ *SD*) |
|:---------:|:------:|:-------------------------:|
| AB | EM | .855 $\pm$ .092 |
| AB | BB | .824 $\pm$ .105 |
| NB | EM | .883 $\pm$ .079 |
| NB | BB | .860 $\pm$ .096 |

- All well above chance (50%)
- **H1 pattern:** NB (*M* = .871) > AB (*M* = .840) (consistent)
- **H2 pattern:** EM (*M* = .870) > BB (*M* = .843) (consistent)

::: notes
Now let us test our hypotheses on accuracy, starting with descriptive statistics. All accuracy values were well above the 50 percent chance level, ranging from 82 to 88 percent. Looking at the H1 pattern, the Natural Boundary group had a mean accuracy of 87.1 percent, compared to 84.0 percent for the Abrupt Boundary group. That is consistent with H1. For H2, Event-Model targets were recognized at 87.0 percent versus 84.3 percent for Boundary-Break targets. That is consistent with H2 as well.
:::

## Accuracy -- Normality Check and Inferential Tests

\begin{columns}
\begin{column}{0.52\textwidth}
\textbf{Normality:} Violated in all 4 cells (\textit{p} < .002) \\
Levene's: variance homogeneity OK (\textit{p} > .08)

\vspace{0.15cm}
\textbf{Parametric (mixed ANOVA):}
\begin{itemize}
  \item H1: \textit{F}(1, 168) = 7.25, \textit{p} = .008, $\eta_p^2$ = .041 (supported)
  \item H2: \textit{F}(1, 168) = 11.44, \textit{p} < .001, $\eta_p^2$ = .064 (supported)
  \item H3: \textit{F}(1, 168) = 0.21, \textit{p} = .651 (n.s.)
\end{itemize}
\textbf{Non-parametric (normality violated):}
\begin{itemize}
  \item H1: Mann-Whitney \textit{U} = 2744, \textit{p} = .007 (supported)
  \item H2: Wilcoxon \textit{W} = 3135, \textit{p} = .003 (supported)
  \item H3: \textit{U} = 3644, \textit{p} = .904 (n.s.)
\end{itemize}
\end{column}
\begin{column}{0.45\textwidth}
\includegraphics[width=\textwidth]{output/fig1_accuracy_barplot.png}
\end{column}
\end{columns}

::: notes
Before running inferential tests, we checked normality. Shapiro-Wilk tests showed normality was violated in all four cells, with all p-values below point-zero-zero-two. Levene's test confirmed homogeneity of variance. Because normality was violated, we ran both parametric and non-parametric tests. The parametric mixed ANOVA showed that H1 was supported: boundary type was significant with F of 7.25 and p equals point-zero-zero-eight. H2 was also supported: target type was significant with F of 11.44 and p less than point-zero-zero-one. H3 was not supported for accuracy: the interaction was not significant. The non-parametric robustness checks confirmed all these results: Mann-Whitney confirmed H1, Wilcoxon confirmed H2, and the interaction remained non-significant.
:::

## Accuracy -- Effect Sizes

- **H1: NB > AB**, *d* = 0.41 (medium effect)
  - NB (*M* = .871) vs. AB (*M* = .840)
- **H2: EM > BB**, *d* = 0.29 (small--medium effect)
  - EM (*M* = .870) vs. BB (*M* = .843)
- **H3: No interaction** (*p* = .651)
  - The EM--BB advantage is similar in both groups

\vspace{0.2cm}

$\rightarrow$ Natural boundaries help memory; EM frames help memory; effects are additive.

Now Hrishiraj will present the RT and confidence results.

::: notes
Let me summarize the effect sizes. For H1, the Natural Boundary group outperformed the Abrupt Boundary group with Cohen's d of 0.41, a medium effect. For H2, Event-Model targets were recognized better than Boundary-Break targets with d of 0.29, a small to medium effect. For H3, there was no interaction, meaning the EM-BB advantage is similar in both groups. So the effects of boundary type and target type on accuracy are additive. Natural boundaries help memory, and Event-Model frames are easier to recognize, independently of each other. Now I will hand over to Hrishiraj for the response time and confidence results.
:::

# RT, Confidence, and Conclusion (Hrishiraj)

## H1, H2, H3: Response Time

\begin{columns}
\begin{column}{0.52\textwidth}
\textbf{Normality:} Violated in all 4 cells (\textit{p} < .002) \\
Levene's: OK (\textit{p} > .08)

\vspace{0.15cm}
\textbf{Descriptive:} RT = 5.45 -- 5.83 s; small differences.

\vspace{0.15cm}
\textbf{Parametric ANOVA:}
\begin{itemize}
  \item H1: \textit{F}(1, 168) = 1.10, \textit{p} = .296 -- n.s.
  \item H2: \textit{F}(1, 168) = 1.16, \textit{p} = .284 -- n.s.
  \item H3: \textit{F}(1, 168) = 3.36, \textit{p} = .069, $\eta_p^2$ = .020
\end{itemize}
\textbf{Non-parametric:} All n.s. (\textit{p} > .09)

\vspace{0.1cm}
Near-significant interaction warrants future investigation.
\end{column}
\begin{column}{0.45\textwidth}
\includegraphics[width=\textwidth]{output/fig2_rt_barplot.png}
\end{column}
\end{columns}

::: notes
Hello, I am Hrishiraj. I will present the response time and confidence results, followed by our conclusions. Starting with response time. Normality was violated in all four cells, so we report both parametric and non-parametric tests. Response times ranged from 5.45 to 5.83 seconds with small differences between conditions. None of the three hypotheses were supported for RT. H1, boundary type, was not significant. H2, target type, was not significant. H3, the interaction, approached significance with p of point-zero-six-nine, which is worth noting for future investigation. Non-parametric tests converged: all non-significant.
:::

## H1, H2, H3: Confidence -- Descriptive + Normality

| Condition | Target | Confidence (*M* $\pm$ *SD*) |
|:---------:|:------:|:---------------------------:|
| AB | EM | 4.13 $\pm$ .46 |
| AB | BB | 4.04 $\pm$ .49 |
| NB | EM | 4.21 $\pm$ .44 |
| NB | BB | 4.20 $\pm$ .42 |

- **H2 pattern:** EM > BB within AB, but **not within NB** (possible interaction)
- **Normality:** 3 cells normal; NB $\times$ EM violated (*p* = .004)
- Levene's: variance homogeneity OK

::: notes
Now for confidence ratings, our most interesting finding. Looking at the descriptive statistics, within the Abrupt Boundary group, EM targets had a mean confidence of 4.13 compared to 4.04 for BB targets, a clear difference. But within the Natural Boundary group, EM was 4.21 and BB was 4.20, virtually no difference at all. This already suggests a possible interaction. For normality, three of four cells were normal, but the NB by EM cell was non-normal with p of point-zero-zero-four. So we will report non-parametric checks alongside the ANOVA.
:::

## H1, H2, H3: Confidence -- Inferential Tests

\begin{columns}
\begin{column}{0.52\textwidth}
\textbf{Parametric ANOVA:}
\begin{itemize}
  \item H1: \textit{F}(1, 168) = 3.24, \textit{p} = .074 -- n.s.
  \item H2: \textit{F}(1, 168) = 5.70, \textit{p} = .018, $\eta_p^2$ = .033 (supported)
  \item \textbf{H3: \textit{F}(1, 168) = 4.03, \textit{p} = .046, $\eta_p^2$ = .023} (supported)
\end{itemize}
\textbf{Simple effects (H3):}
\begin{itemize}
  \item AB: EM > BB (\textit{p} = .003, \textit{d} = .19)
  \item NB: EM $\approx$ BB (\textit{p} = .733) -- no difference
  \item BB: NB > AB (\textit{p} = .024, \textit{d} = .35)
\end{itemize}
\textbf{Non-parametric:} Wilcoxon confirmed H2 (\textit{p} = .025). Mann-Whitney confirmed H3 (\textit{p} = .025).
\end{column}
\begin{column}{0.45\textwidth}
\includegraphics[width=\textwidth]{output/fig7_confidence_interaction.png}
\end{column}
\end{columns}

::: notes
The mixed ANOVA on confidence revealed the following. H1 was not significant for confidence, with p of point-zero-seven-four. H2 was significant: EM targets received higher confidence than BB targets, with p of point-zero-one-eight. Most importantly, H3 was supported: there was a significant interaction with p of point-zero-four-six. Looking at simple effects, within the Abrupt Boundary group, confidence was significantly higher for EM than BB targets with d of 0.19. But within the Natural Boundary group, there was no difference at all. For BB targets specifically, the Natural Boundary group was more confident than the Abrupt Boundary group with d of 0.35. Non-parametric Wilcoxon confirmed H2, and Mann-Whitney confirmed H3, including the simple effects pattern.
:::

## Key Finding: Accuracy vs. Confidence Dissociation

\vspace{0.2cm}

| | H1 (Boundary) | H2 (Target) | H3 (Interaction) |
|:---|:---:|:---:|:---:|
| **Accuracy** | Significant | Significant | **Not significant** |
| **Confidence** | Not significant | Significant | **Significant** |

\vspace{0.2cm}

- Accuracy: both main effects, **no interaction** (effects are additive)
- Confidence: **significant interaction** (effects are not additive)
- AB group is less confident about BB targets, but NB group is equally confident
- Abrupt boundaries **selectively impair metacognitive certainty** for boundary-adjacent content

::: notes
The key finding of our study is the dissociation between accuracy and confidence. Look at this table. For accuracy, both H1 and H2 were significant, but H3, the interaction, was not. The effects of boundary type and target type on accuracy are additive. But for confidence, H2 and H3 were significant, meaning the interaction was present. This means that abrupt boundaries selectively impair metacognitive certainty for boundary-adjacent content. The Abrupt Boundary group recognized BB frames almost as well as EM frames, but they were significantly less confident about those BB judgments. The Natural Boundary group showed stable confidence regardless of frame type. This is something that accuracy alone would not reveal.
:::

## Correlations and Summary

**Spearman correlations (between DVs):**

- Accuracy--Confidence: $\rho$ = 0.360, *p* < .0001 (metacognitive calibration)
- RT--Confidence: $\rho$ = --0.152, *p* = .047 (faster = more confident)
- Accuracy--RT: $\rho$ = --0.054, *p* = .487 (no speed--accuracy tradeoff)

\vspace{0.2cm}

**Summary of hypothesis tests:**

- **H1 supported** for accuracy (*d* = 0.41); NB > AB
- **H2 supported** for accuracy (*d* = 0.29) and confidence (*d* = 0.11); EM > BB
- **H3 supported** for confidence only: AB group less confident about BB targets
- All conclusions robust to non-parametric testing

::: notes
We also examined Spearman correlations. Accuracy and confidence were positively correlated, indicating reasonable metacognitive calibration. RT and confidence were weakly negatively correlated. Accuracy and RT were not correlated, ruling out a speed-accuracy tradeoff. To summarize our hypothesis tests: H1 was supported for accuracy with a medium effect size. H2 was supported for both accuracy and confidence. H3, the interaction, was supported for confidence but not accuracy, revealing the dissociation between what people get right and how certain they feel about it. All conclusions were robust to non-parametric testing, giving us confidence despite the normality violations.
:::

## Limitations and Future Directions

- No demographic data captured (no individual difference analyses)
- Item-level variability not modelled
  - Future: mixed-effects models with crossed random effects
- Signal Detection Theory (*d'*, criterion *c*) could add nuance
- Near-significant RT interaction (*p* = .069) warrants follow-up
- **All non-parametric tests confirmed parametric conclusions**

::: notes
We should note some limitations. Demographic information was not captured, preventing individual difference analyses. Item-level variability was not modelled. Future work could use mixed-effects models. Signal Detection Theory measures would add nuance. The near-significant RT interaction deserves follow-up with a larger sample. On the positive side, all non-parametric robustness checks fully confirmed our parametric conclusions, giving us strong confidence in every result we reported.
:::

## Thank You

\centering

**Team Odomos**

\vspace{0.3cm}

Archit Choudhary | Bhavya Ahuja | Hrishiraj Mitra

\vspace{0.5cm}

**Questions?**

::: notes
Thank you for your attention. We are happy to take any questions.
:::

## References

\small

- Swallow, K. M., Zacks, J. M., \& Abrams, R. A. (2009). Event boundaries in perception affect memory encoding and updating. *Journal of Experimental Psychology: General*, *138*(2), 236--257.

- Zacks, J. M., \& Swallow, K. M. (2007). Event segmentation. *Current Directions in Psychological Science*, *16*(2), 80--84.

- Zacks, J. M., Speer, N. K., Swallow, K. M., Braver, T. S., \& Reynolds, J. R. (2007). Event perception: A mind--brain perspective. *Psychological Bulletin*, *133*(2), 273--293.

::: notes
These are the references for our study.
:::
