# Project Report — Income Classification & Customer Segmentation

A central goal of most businesses is to better understand their customers: who they are, what they need, and how to serve them effectively. In today’s business environment, this requires a strategic and systematic approach to customer analysis, often supported by business intelligence methods. In this project, we conducted a structured analysis of potential customers to identify signals related to wealth level and customer behavior. Based on these findings, we developed customer profiles that can help group individuals with similar characteristics and support more targeted marketing strategies.

## 1. Project objective

To better assist our retail client who aim to identify two groups for marketing purposes: people earning above and below \$50K annually, we have build two models to achieve this goal:

1. A **classifier** that predicts which side of the \$50K threshold a given person falls on, trained and validated on labeled census data.
2. A **segmentation model** that groups the broader population into actionable marketing groups.

The goal here is targeted marketing, not credit or eligibility decisions. The classifier produces a ranking the client can use to prioritize prospects; the segmentation tells the client what kind of person is in each segment, so different campaigns can be tailored to different groups. The two outputs are complementary to each other and will better assist decision making: classifier picks the targets, segmentation decides the messaging.

The dataset is the 1994–1995 Current Population Survey (Census-Income KDD), roughly 200K rows with 40 demographic and employment variables plus a sampling weight. The dataset's labels are provided, making it a supervised problem; the question is how well the patterns in 1994–1995 generalize to a modern client, which is addressed in the Limitations section.

## 2. Data exploration

The exploration notebook (`0_eda.ipynb`) covered five things: data quality, target distribution, feature-target relationships, redundancy across features, and a feature importance prior via mutual information. A few findings shaped everything downstream.

**Class imbalance is heavy.** The positive class (>\$50K) is roughly 6% of the sample, and the weighted prevalence (after applying the sample weight) differs from the unweighted version. That difference is itself a finding: the survey slightly over-represents the high-income class relative to the population, which is why the sample weight is adopted in evaluation and in model fitting.

**Missing values are encoded in multiple ways.** Beyond standard `NaN`, several tokens act as sentinels: `?`, `Not in universe`, `Not in universe or children`, `Not in universe under 1 year old`, and `Do not know`. These are not equivalent. `?` is item non-response (the question applied but went unanswered); the `Not in universe` family encodes skip logic (the question didn't apply at all). This set of sentinel terms were derived programmatically by enumerating values that appear across many object columns, then filtered out false positives that contained "not in" but were meaningful category labels (e.g., household-relationship strings).

**Three financial columns are heavily zero-inflated.** Capital gains, capital losses, and dividends from stocks are zero for ~95% of rows, but the nonzero subgroup has a positive rate substantially above baseline. This pattern motivated an engineered `has_nonzero` binary flag and a `log1p` transform before scaling in the preprocessing step.

**Migration columns are administratively year-determined.** The Cramér's V analysis identified a perfect dependence (V = 1.0) between `year` and five migration columns. Investigation showed the migration questions were administered differently across the two cohorts from 1994 and 1995, with one year carrying mostly `Not in universe` responses. These columns provide no useful signal for an income classifier and were dropped before modeling.

**Industry and occupation are encoded twice.** Detailed and major versions exist for both; the detailed versions are numeric codes, the major versions are readable category labels. I kept the detailed versions and dropped the major ones to preserve information richness.

**Mutual information ranking confirmed intuition.** Dividends from stocks, capital gains, education, marital status, and occupation dominate the top of the MI ranking. Several features at the bottom (year, migration variables, hispanic origin, own business indicator) carry near-zero univariate signal — most were dropped, though `hispanic origin` and `own business or self employed` were retained for fairness reporting and interpretability respectively.

**The strongest individual signals are unsurprising but worth quantifying.** Against a baseline positive rate of ~6.2%:

- Education shows a clean monotonic pattern: professional school degrees 54%, doctorate 52%, masters 31%, bachelors 20%, high school 4%, less than high school near 0%. This is the cleanest single feature in the data and motivated the `education_ordinal` engineering choice.
- Occupation: executive/managerial 29%, professional specialty 25% — both 4–5× baseline.
- Class of worker: self-employed-incorporated 35% — by far the highest of any worker class, motivating the `is_self_employed` flag.
- Sex: men 10.2%, women 2.6% — a 4× gap that foreshadows the fairness findings in Section 7.
- Marital status: married-civilian-spouse-present 11.4% vs. never married 1.3% — motivates the `married_joint_filer` engineering choice.
- Tax filer status: joint filers under 65 13.1%, nonfilers ~0% — joint filing is a strong household-level proxy.

## 3. Preprocessing and feature engineering

The preprocessing notebook (`1_preprocessing.ipynb`) applies the cleanup and engineering steps that the EDA findings motivated.

**Loading discipline.** The raw file is read with `dtype=str` and whitespace stripped per cell. Numerics are then cast explicitly with `errors="raise"` so silent type coercion is impossible. Integer-coded categoricals (veteran benefits, year, etc.) are kept as strings for they carry semantic meaning, and to prevent a linear model from interpreting their codes as ordinal magnitudes.

**Drops.** Migration columns and `year` removed for the reason above. Major industry/occupation recodes removed in favor of the detailed versions. `wage per hour` removed because it's structurally absent for salaried and self-employed workers, and the engineered `is_employed` flag captures the labor-force participation signal more cleanly. MI table also confirmed that this column provided limited information. 

**Engineered features.** Five features added to bridge the gap between what trees can learn natively and what the logistic baseline can express:

- `education_ordinal`: ordinal encoding of the 17 education levels, so the linear model can express "more education → higher income" as a single coefficient instead of 17 dummies.
- `n_income_sources`: count (0–3) of earned-income presence, capital gains presence, and dividend presence. Surfaces the wealth-diversity signal that the linear model can't otherwise learn as an AND interaction.
- `is_employed`: binary derived from `weeks worked > 0`. Cleanly separates the structural "in labor force" signal that's otherwise fragmented across multiple columns. 
- `is_self_employed`: binary from `class of worker`. Self-employed have a qualitatively different income profile (lower wages, higher investment income) and surfacing this makes the interaction with the wealth features more legible.
- `married_joint_filer`: binary from `tax filer stat`. Proxy for two-earner households, since the dataset doesn't have a direct spouse-income field.

**Missing value handling.** `?` replaced with `NaN` and imputed by mode for categoricals. NIU variants retained as their own category, since "the question didn't apply" is itself informative.

**Categorical encoding.** Low-cardinality (≤15 unique values) features one-hot encoded. High-cardinality features target-encoded using sklearn's `TargetEncoder` with built-in cross-validation to prevent leakage.

**Numeric transforms.** Capital gains, capital losses, and dividends transformed with `log1p` before scaling — preserves zeros at zero while compressing the heavy tail. Other numerics scaled with `StandardScaler`.

**Train/test split.** Stratified 80/20 random split rather than temporal (1994 → 1995). Cross-sectional survey data with consistent methodology across two consecutive years is reasonably independent; the migration-column distribution shift that would have motivated a temporal split was already removed by dropping those columns. A temporal split could be further explored in the future as a robustness check appropriate for production deployment.

## 4. Modeling approach

The classification notebook (`2_classifier.ipynb`) trains three models for comparison: a logistic regression baseline, XGBoost, and LightGBM.

**Baseline.** Logistic regression with `class_weight='balanced'` (for class imbalance) and `sample_weight=w_train` (for population alignment with the survey weights). This model serves as the baseline for comparison, thus not tuned.

**Gradient-boosted models.** Both XGBoost and LightGBM are tuned with `RandomizedSearchCV`: 20 iterations, 3-fold stratified cross-validation, `average_precision` (PR-AUC) as the scoring metric. Search space covered tree complexity, learning rate, subsampling, and L1/L2 regularization. After tuning, models refit on the full training data with early stopping (50 rounds, empirical) against a 20% validation slice carved from the training data.

**Two weighting mechanisms, two purposes.** Throughout training and evaluation, `sample_weight=w` scales the loss function with the population the sample was designed to represent. `scale_pos_weight` (in XGB and LGB) and `class_weight='balanced'` (in the baseline) separately handle the class imbalance. These are different concerns and both are needed.

**Model selection.** Final model chosen on population-weighted test PR-AUC. Selecting on test is acceptable because no further model changes are made after selection.

The segmentation notebook (`3_segmentation.ipynb`) uses K-means on the population, with population-weighted centroids via `sample_weight`. Seven interpretable features were selected: age, education_ordinal, weeks worked, is_employed, is_self_employed, married_joint_filer, is_male. Wealth indicators (capital gains, dividends, n_income_sources) were deliberately excluded so the segmentation surfaces demographic structure rather than rediscovering the classifier's decision boundary. Race and country-of-origin were excluded to avoid using sensitive attributes as primary clustering axes in a marketing-targeting context.

K was selected by joint inspection of inertia and silhouette score across K=2 through 8. The statistical optimum was K=3, but K=4 was selected because it produced a more useful set of personas: K=3 collapsed students and established professionals into the same cluster, while K=4 separated them into distinct personas. The interpretability gain justified the small silhouette penalty.

The fairness notebook (`4_fairness.ipynb`) maps the protected attributes (sex, race, hispanic origin, citizenship) to the test set and computes selection rate, disparate impact ratio, TPR, FPR, and precision by group at the F1-optimal threshold. Segment-level demographic composition is also reported.

## 5. Model results and evaluation

LightGBM was selected as the production model based on the population-weighted PR-AUC on the held-out test set.

| Model | ROC-AUC (population) | PR-AUC (population) |
|---|---|---|
| Logistic Regression | 0.948 | 0.636 |
| XGBoost (tuned) | 0.954 | 0.689 |
| LightGBM (tuned) | **0.955** | **0.691** |

PR-AUC is the headline metric: with ~6% positive class, the random baseline PR-AUC equals the prevalence, so the LightGBM model is roughly 11× better than random by this measure. ROC-AUC is reported alongside for comparability but is less sensitive to imbalance.

**Lift and gain.** Translated into the marketing use case, the model is very strong:

- The top decile by score (top 10% of prospects) captures roughly 77% of the high-income population. Lift over random targeting in this decile is ~7.5×.
- Expanding the targeting to the top 20% raises capture to ~92% of the high-income population.
- The bottom three deciles contain essentially zero high-income individuals — the model is highly effective at excluding obvious non-targets.

The gain chart shows the full tradeoff curve. The asymptotic shape (sharp rise then quick plateau) is characteristic of a strong classifier on heavily imbalanced data.

**Operating points.** Two thresholds are reported:

- F1-optimal threshold (~0.85): precision 0.64, recall 0.62. Best single classifier balance.
- Top-20% campaign threshold (~0.39): precision 0.29, recall 0.92. Population-weighted. For a moderate marketing budget, the campaign reaches ~90% of the high-income population at a ~31% conversion rate (vs. ~6% baseline if untargeted).

**Calibration.** The model's raw probabilities are not well-calibrated because `scale_pos_weight` rebalancing inflates predicted probabilities. This is why the F1-optimal threshold is 0.85 rather than 0.5. Rankings (which is what marketing targeting uses) are reliable. For any application that depends on the absolute probability score, `CalibratedClassifierCV` with isotonic regression would be needed.

**Feature importance.** SHAP values on a 2,000-row test sample. Based on the EDA findings, the top contributors should include dividends from stocks, capital gains, `education_ordinal`, major occupation code (executive/managerial and professional specialty being the strongest categories), marital status, age, sex, and several of the engineered flags (`n_income_sources`, `married_joint_filer`, `is_self_employed`). age, education level, tax filer stat, occupation code, weeks worked in year, sex, source of income, etc. 

## 6. Segmentation results

Four segments at K=4. The table below shows population share, high-income rate within the segment, share of the total high-income population each segment contains, and targeting efficiency (share of high-income / share of population — values above 1 indicate over-representation).

| Segment | Persona | Population Share | High-Income Rate | Share of High-Income | Targeting Efficiency |
|---|---|---|---|---|---|
| 1 | Experienced Professionals | 45.4% | 13.3% | 94.0% | 2.07× |
| 2 | Retirees | 15.8% | 2.1% | 5.2% | 0.33× |
| 0 | Early Career / Students | 15.1% | 0.3% | 0.7% | 0.05× |
| 3 | Children / Minors | 23.7% | 0.0% | 0.0% | 0.00× |

The concentration of the high-income population in a single segment is extreme: **94% of all >$50K individuals fall in Segment 1 (Experienced Professionals)**, even though that segment is only 45% of the population. Targeting efficiency 2.07× means a marketing campaign sent only to this segment would already get a 2× lift over random outreach, before any classifier scoring on top.

The remaining three segments are functionally non-targets for premium-product marketing:

- **Retirees** carry 5% of the high-income population at a within-segment rate of 2.1%. Not zero, but well below baseline. Targeting efficiency 0.33×.
- **Early Career / Students** has effectively no current high-income members (0.3% rate). The persona's value to the client is future income, not present.
- **Children / Minors** has zero high-income individuals by definition. This segment exists in the data because the CPS samples entire households; it's a structural artifact rather than a marketing audience.

**Marketing-strategy takeaway.** The classifier's lift comes from finding the 13% of Segment 1 members who are above the income threshold. The segmentation tells the client *where* to look — campaigns aimed at Segment 1 will be far more efficient than campaigns spread across the full population, even before applying classifier scoring. Combining the two: target prospects with high classifier scores *within* Segment 1, and refine messaging by what Segment 1 actually contains (working-age professionals with employment and family signals).

## 7. Fairness analysis

An adverse-impact analysis was performed at the F1-optimal threshold across four protected attributes: sex, race, Hispanic origin, and citizenship. For each, the table below shows the selection rate (fraction of the group the model would target at the operating threshold) and the disparate impact ratio (DIR) relative to the group with the highest selection rate. The four-fifths rule flags any DIR below 0.80 as a potential adverse impact.

**Sex.**

| Group | Selection rate | DIR |
|---|---|---|
| Male | 0.115 | 1.000 |
| Female | 0.018 | 0.154 |

The sex disparity is severe. Women would be targeted at roughly 15% the rate of men. This is the largest single disparity in the analysis.

**Race.**

| Group | Selection rate | DIR |
|---|---|---|
| Asian or Pacific Islander | 0.078 | 1.000 |
| White | 0.073 | 0.937 |
| Amer Indian Aleut or Eskimo | 0.046 | 0.595 |
| Black | 0.016 | 0.201 |
| Other | 0.014 | 0.181 |

White passes the four-fifths rule. American Indian / Alaska Native falls below it (DIR 0.595). Black and Other subgroups are severely under-selected at DIR ~0.18–0.20 — less than a quarter of the rate the model targets the Asian/Pacific Islander reference group.

**Hispanic origin.**

| Group | Selection rate | DIR |
|---|---|---|
| All other (non-Hispanic) | 0.071 | 1.000 |
| Do not know | 0.051 | 0.715 |
| Other Spanish | 0.029 | 0.413 |
| Cuban | 0.028 | 0.394 |
| Chicano | 0.018 | 0.260 |
| Puerto Rican | 0.018 | 0.255 |
| Mexican-American | 0.015 | 0.213 |
| Central or South American | 0.014 | 0.204 |
| Mexican (Mexicano) | 0.006 | 0.091 |

Every Hispanic-origin subgroup falls below the 80% rule, most of them substantially. The Mexican (Mexicano) subgroup has a DIR of 0.091 — members of this group are selected at less than a tenth of the rate of the non-Hispanic reference group.

**Citizenship.**

| Group | Selection rate | DIR |
|---|---|---|
| Foreign born — US citizen by naturalization | 0.100 | 1.000 |
| Native born abroad of American parent(s) | 0.069 | 0.688 |
| Native — Born in the United States | 0.066 | 0.664 |
| Foreign born — Not a citizen of US | 0.035 | 0.346 |
| Native — Born in Puerto Rico or US Outlying | 0.018 | 0.177 |

A counterintuitive pattern here: naturalized citizens are selected at the highest rate of any citizenship group, higher than native-born US citizens. This likely reflects selection effects in the naturalization process — people who complete the lengthy citizenship process tend to be economically integrated. The lowest rates are for non-citizens (DIR 0.346) and people born in Puerto Rico or US outlying areas (DIR 0.177), both severely below the 80% threshold.

**Interpretation.** The disparities are real and substantial. They are not necessarily "model bias" in a narrow sense — the classifier is approximating patterns in the labeled  data, and that data reflects the actual income distribution of 1994–1995, which itself reflects the gender wage gap, racial wealth gap, and immigration-status earning disparities of the era. The model would propagate these historical patterns at scale if deployed for marketing targeting. For a marketing application this carries reputational and fair-marketing risk. For any application closer to lending, eligibility, or pricing — which this is not, but related models often become — these patterns would also carry regulatory risk under the four-fifths rule.

**Recommended pre-deployment actions.**

1. **Do not deploy as-is.** Multiple protected attributes show DIRs well below 0.80 — the sex disparity at DIR 0.154 alone would require mitigation under any standard adverse-impact framework, and several race, Hispanic-origin, and citizenship subgroups are at or below DIR 0.20.
2. **Audit the actual targeting list against the client's stated audience goals** before any campaign launch. The demographic composition of a top-20% campaign should be reviewed by the client's marketing and compliance teams together.
3. **Consider fairness-aware re-training.** Reweighting, equalized-odds post-processing, or constrained optimization can be applied without changing the underlying model architecture. Fairlearn and similar libraries provide well-tested implementations.
4. **Constrain targeting to be balanced within high-scoring deciles.** Instead of taking the top 20% by raw score, take the top 20% within each protected-attribute subgroup. This trades some classifier efficiency for parity across groups.
5. **Establish monitoring post-launch.** Track selection rates by protected group over time, with alerts if any group drifts further below threshold as the model is retrained on newer data.

None of these were implemented within this project's scope; they are recommended as pre-deployment requirements rather than present remediations.

## 8. Business judgment and recommendations

The recommendations below assume the fairness disparities documented in Section 7 are addressed before any production deployment. The model is technically strong but is not deployment-ready as-is.

**Recommended usage.** Use the classifier to rank prospects by predicted score, then use the segmentation to choose the campaign for each prospect. The classifier answers "who?", the segmentation answers "what to send them?". The two outputs are complementary.

**Recommended operating point.** Top-20% targeting (threshold ~0.38) hits the sweet spot for a moderate marketing budget: ~90% capture of the high-income population at a 5× efficiency lift over random outreach. For a tighter budget, top-10% targeting still captures ~77% of high-income individuals at ~7.5× lift. Both operating points should be modified by the fairness mitigation chosen — for example, taking the top 20% within each protected subgroup rather than the top 20% overall.

**Most actionable segment.** Concentrate campaigns on Segment 1 (Experienced Professionals) — this segment contains 94% of the high-income population while being only 45% of the total population. The classifier should be applied within this segment rather than across the full population for most campaign types. Segments 0, 2, and 3 (Early Career, Retirees, Children) collectively contain only 6% of high-income individuals and should be addressed with different campaigns or excluded from premium-product targeting entirely.

## 9. Limitations and next steps

**Data vintage.** The CPS data is from 1994–1995. Income distributions, occupation prevalences, and demographic compositions have shifted substantially since then. The model and segmentation should be updated on current data before any production deployment. The methodology described here is reusable; the specific coefficients and segment boundaries are not.

**Probability calibration.** The classifier's raw probabilities are inflated due to `scale_pos_weight`. Rankings are reliable; absolute probabilities are not. If the client needs probabilistic outputs (e.g., expected revenue computations), apply isotonic calibration.

**Fairness as a pre-deployment requirement, not a present remediation.** The fairness analysis identifies disparities but does not fix them. Auditing and mitigation must happen before campaign launch.

**Temporal validation skipped.** The model was validated by stratified random split rather than a 1994 → 1995 temporal split. The migration columns (the main source of year-to-year distribution shift in this dataset) were already dropped, so the temporal robustness gain would have been modest, but a year-out validation is a reasonable robustness check for production.

**Next steps.**

1. Re-train and re-segment on current customer data before any campaign launch.
2. Apply isotonic calibration if the client needs interpretable probabilities.
3. Conduct the fairness audit and apply mitigation as needed.
4. Run a controlled A/B test on a small pilot campaign before scaling.
5. Establish monitoring on selection rates by protected group post-launch to catch drift.

## 10. References and resources

**Dataset and survey methodology**

- Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository: Census-Income (KDD) Data Set.* University of California, Irvine.
- U.S. Census Bureau, *Current Population Survey: Design and Methodology* (technical paper, 1994–1995 releases). Source for the survey's stratified sampling design and the meaning of the instance weight.

**Statistical and machine learning methods**

- https://scikit-learn.org/stable/. Usage, grammar and parameters of scikit-learn. 
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research. Used for the logistic baseline, preprocessing pipeline (`TargetEncoder`, `StandardScaler`, `ColumnTransformer`), `RandomizedSearchCV`, K-means, and calibration utilities.
- Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of KDD '16. Foundation for the gradient-boosted classifier; documentation referenced for `scale_pos_weight`, early stopping, and PR-AUC objective.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017. Referenced for leaf-wise tree construction and the `num_leaves` parameterization used in tuning.
- Lundberg, S. M. and Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. Source for SHAP and `TreeExplainer` used in the feature-importance analysis.

**Fairness and adverse-impact analysis**

- U.S. Equal Employment Opportunity Commission. *Uniform Guidelines on Employee Selection Procedures (29 CFR 1607).* Source for the four-fifths rule used in the disparate impact analysis.
- Barocas, S., Hardt, M., and Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities.* fairmlbook.org. Background reading for the framing of fairness metrics (selection rate, equal opportunity, demographic parity).
- Bird, S. et al. (2020). *Fairlearn: A toolkit for assessing and improving fairness in AI.* Microsoft Research. Referenced as the recommended tooling for the fairness-aware retraining listed in the pre-deployment actions.


