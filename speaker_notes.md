# Speaker Notes: Event Boundaries and Movie Memory
## 15-Minute Presentation, Team Odomos

---

## Part 1: Archit Choudhary (Slides 1 to 4, approximately 5 minutes)

### Slide 1: Background -- Event Segmentation Theory (1.5 minutes)

Hello everyone. I am Archit, and I will introduce our study on how event boundaries in movies influence memory. Event Segmentation Theory, proposed by Zacks and colleagues in 2007, tells us that our brains automatically segment continuous experience into discrete events. The transitions between these events, called event boundaries, trigger an update of our internal event model. This is a real-time mental representation of the current situation. This updating process has direct consequences for memory: content near event boundaries is encoded differently from mid-event content, as shown by Swallow, Zacks, and Abrams in 2009. Now, there are two types of event boundaries. Natural boundaries are smooth, organic transitions between activities. Abrupt boundaries are sudden, artificially imposed transitions, like hard cuts in a film.

### Slide 2: Task Paradigm (1.5 minutes)

This figure shows our experimental paradigm. In the encoding phase, participants watched 40 short movie clips. One group, the AB group, saw clips with abrupt hard cuts at event boundaries. The other group, the NB group, saw clips with smooth, natural transitions. After a delay, all participants completed a two-alternative forced-choice recognition test with 40 trials. On each trial, they saw two frames: a target they had actually seen and a perceptually similar lure. They had to identify the real frame, then rate their confidence on a 1 to 5 scale. Target frames were of two types. Event-Model frames were consistent with the ongoing event representation. Boundary-Break frames were drawn from near an event boundary. We measured three dependent variables: accuracy, response time, and confidence.

### Slide 3: Design and Participants (30 seconds)

We used a two-by-two mixed design. Boundary type was between-subjects and target frame type was within-subjects. We had 170 participants, 81 in the Abrupt Boundary group and 89 in the Natural Boundary group, after excluding one for missing data. Each participant did 40 trials, giving us 6,800 trials total.

### Slide 4: Hypotheses (1.5 minutes)

We had three hypotheses, and we tested each one across all three dependent variables. Hypothesis 1: the Natural Boundary group will show better recognition performance than the Abrupt Boundary group, because natural event segmentation supports more coherent encoding. Hypothesis 2: Event-Model targets will be recognized more accurately and with greater confidence than Boundary-Break targets, because Event-Model frames are consistent with the maintained event representation. Hypothesis 3: the advantage of Event-Model over Boundary-Break targets may be larger in the Abrupt Boundary group, because abrupt boundaries may differentially impair encoding of boundary-adjacent content. With that, I will hand it over to Bhavya for the methods and accuracy results.

---

## Part 2: Bhavya Ahuja (Slides 5 to 8, approximately 5 minutes)

### Slide 5: Data Processing and Analytical Pipeline (1.5 minutes)

Hello, I am Bhavya. Let me describe our methods. We parsed 171 PsychoPy CSV files. One participant was excluded, leaving 170 participants and 6,800 trials. From each trial we extracted accuracy, response time, and confidence. One RT trial was removed as an outlier. For each dependent variable, we followed a systematic four-step pipeline. Step one: compute descriptive statistics. Step two: check normality using Shapiro-Wilk tests on each cell plus Levene's test for variance homogeneity. Step three: run a two-by-two mixed ANOVA with partial eta-squared, and follow up significant effects with t-tests reporting Cohen's d. Step four: if normality was violated, run non-parametric robustness checks. Mann-Whitney U for between-subjects effects, Wilcoxon signed-rank for within-subjects effects, and Mann-Whitney on difference scores for the interaction.

### Slide 6: Accuracy -- Descriptive Statistics (1 minute)

Now let us test our hypotheses on accuracy, starting with descriptive statistics. All accuracy values were well above the 50 percent chance level, ranging from 82 to 88 percent. Looking at the H1 pattern, the Natural Boundary group had a mean accuracy of 87.1 percent, compared to 84.0 percent for the Abrupt Boundary group. That is consistent with H1. For H2, Event-Model targets were recognized at 87.0 percent versus 84.3 percent for Boundary-Break targets. That is consistent with H2 as well.

### Slide 7: Accuracy -- Normality and Inferential Tests (1.5 minutes)

Before running inferential tests, we checked normality. Shapiro-Wilk tests showed normality was violated in all four cells, with all p-values below point-zero-zero-two. Levene's test confirmed homogeneity of variance. Because normality was violated, we ran both parametric and non-parametric tests. The mixed ANOVA showed that H1 was supported: boundary type was significant with F of 7.25, p equals point-zero-zero-eight, and partial eta-squared of point-zero-four-one. H2 was also supported: target type was significant with F of 11.44, p less than point-zero-zero-one, partial eta-squared of point-zero-six-four. H3 was not supported for accuracy: the interaction was not significant. The non-parametric tests confirmed all of this: Mann-Whitney confirmed H1, Wilcoxon confirmed H2, and the interaction stayed non-significant.

### Slide 8: Accuracy -- Effect Sizes (1 minute)

For the effect sizes: H1, the Natural Boundary group outperformed the Abrupt Boundary group with Cohen's d of 0.41, a medium effect. H2, Event-Model targets were recognized better with d of 0.29, small to medium. H3, no interaction, meaning the effects are additive. Natural boundaries help memory, and Event-Model frames are easier to recognize, independently of each other. Now I will hand over to Hrishiraj for response time and confidence.

---

## Part 3: Hrishiraj Mitra (Slides 9 to 15, approximately 5 minutes)

### Slide 9: Response Time Results (1 minute)

Hello, I am Hrishiraj. Starting with response time. Normality was violated in all four cells, so we report both parametric and non-parametric tests. Response times ranged from 5.45 to 5.83 seconds with small differences. None of the three hypotheses were supported for RT. H1, boundary type, was not significant. H2, target type, was not significant. H3, the interaction, approached significance with p of point-zero-six-nine, which is worth noting for future investigation but did not reach our alpha of point-zero-five. Non-parametric tests confirmed: all non-significant.

### Slide 10: Confidence -- Descriptive and Normality (1 minute)

Now for confidence, our most interesting finding. Looking at the descriptive statistics, within the Abrupt Boundary group, Event-Model targets had a mean confidence of 4.13 compared to 4.04 for Boundary-Break targets, a clear gap. But within the Natural Boundary group, Event-Model was 4.21 and Boundary-Break was 4.20, virtually identical. This already suggests a possible interaction. For normality, three of four cells were normal but the NB by EM cell was non-normal with p of point-zero-zero-four. So we report non-parametric checks alongside the ANOVA.

### Slide 11: Confidence -- Inferential Tests (1 minute)

The mixed ANOVA on confidence revealed the following. H1 was not significant with p of point-zero-seven-four. H2 was significant: Event-Model targets received higher confidence with p of point-zero-one-eight. Most importantly, H3 was supported: the interaction was significant with p of point-zero-four-six. Simple effects showed that within the Abrupt Boundary group, confidence was significantly higher for Event-Model than Boundary-Break targets with d of 0.19. But within the Natural Boundary group, there was no difference at all. For Boundary-Break targets specifically, the Natural Boundary group was more confident than the Abrupt Boundary group with d of 0.35. Non-parametric tests confirmed H2 and H3 including the simple effects pattern.

### Slide 12: Key Finding -- Accuracy vs. Confidence Dissociation (1 minute)

This is our key finding: the dissociation between accuracy and confidence. For accuracy, H1 and H2 were significant, but H3, the interaction, was not. The effects are additive. For confidence, H2 and H3 were significant, meaning there is an interaction. Abrupt boundaries selectively impair metacognitive certainty for boundary-adjacent content. The Abrupt Boundary group recognized Boundary-Break frames almost as well as Event-Model frames, but they were significantly less confident about those Boundary-Break judgments. The Natural Boundary group showed stable confidence regardless of frame type. This is something that accuracy alone would not reveal.

### Slide 13: Correlations and Summary (30 seconds)

Briefly on correlations: accuracy and confidence were positively correlated, confirming metacognitive calibration. There was no speed-accuracy tradeoff. To summarize: H1 supported for accuracy, H2 supported for accuracy and confidence, H3 supported for confidence only, revealing the dissociation. All conclusions were robust to non-parametric testing.

### Slide 14: Limitations and Future Directions (30 seconds)

Our limitations: no demographic data, no item-level modelling, and the near-significant RT interaction deserves follow-up. Importantly, every non-parametric test confirmed the parametric conclusions.

### Slide 15: Thank You (30 seconds)

Thank you for your attention. We are happy to take any questions.

---

## Timing Summary

| Presenter | Slides | Duration |
|:----------|:------:|:--------:|
| Archit Choudhary | 1 to 4 | 5 min |
| Bhavya Ahuja | 5 to 8 | 5 min |
| Hrishiraj Mitra | 9 to 15 | 5 min |
| **Total** | **15** | **15 min** |
