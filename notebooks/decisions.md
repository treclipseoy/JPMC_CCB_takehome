# Decisions

Notes on non-obvious choices made throughout the project: data type casts, preprocessing (sentinels, missing values, encoding), feature selection, model choices, and evaluation metrics. Anything that isn't a pandas/sklearn default should have an entry here pointing back to the cell where it happened.

### 0_eda.ipynb

1. **(Cell 6) Loading strategy.** Read everything as strings first (`dtype=str`) and strip whitespace cell-by-cell. The file has spaces after commas, and letting pandas guess types on top of that was getting unreliable. Numerics get cast explicitly in the next cell with `errors="raise"` so anything unexpected blows up loudly instead of silently coercing.

2. **(Cell 7) Numeric columns.** Cast seven columns to numeric: `age`, `wage per hour`, `capital gains`, `capital losses`, `dividends from stocks`, `num persons worked for employer`, `weeks worked in year`. Picked these by reading the column names and looking at `df.head()` — they're the ones that actually carry a quantitative measure. Several other integer-looking columns (industry/occupation codes, veteran benefits, year) are kept as strings, since the integers there are codes, not magnitudes. Treating "veteran benefits code 2" as 2x of "code 1" would be nonsense, and that's exactly what a linear model would assume if I cast them.

3. **(Cell 7) Weight column.** `weight` is a sampling weight from the survey's stratified design, not a feature. Dropped from training. Keeping it around for two things later: weighted evaluation metrics, and population-scaled segment sizes when I get to clustering.

4. **(Cell 8) Label normalization.** Mapped the raw label strings (`'- 50000.'`, `'50000+.'`) to 0/1 with an explicit dict. Used a dict instead of a regex on purpose: if some value I didn't expect shows up, it becomes NaN and I'll catch it immediately on a value-count check. A regex might silently match the wrong thing.

5. **(Cell 8) Class imbalance.** Computed positive-class prevalence two ways — unweighted, and weighted by the instance weight. They don't match, which is itself a sampling-design finding worth surfacing in the report. The imbalance is also heavy enough that accuracy is going to be uninformative, so I'm planning to lead with PR-AUC, ROC-AUC, and a decile lift table.

6. **(Cell 11) NIU token derivation.** Ran a regex scan over every object column to find candidate sentinel tokens. The regex turned out to be too greedy: it matched a bunch of household-relationship strings (`'Child 18+ ever marr Not in a subfamily'` etc.) that contain "not in" but are real category labels, not non-responses. After filtering those out by inspection, the actual sentinel set is `{"?", "Not in universe", "Not in universe or children", "Not in universe under 1 year old", "Do not know"}`. One thing worth flagging: `?` and the `Not in universe` family don't mean the same thing. `?` means the question applied and the respondent didn't answer; `Not in universe` means the question didn't apply at all (survey skip logic). I'll impute the `?` rows in preprocessing but keep the NIU variants as a real category, since "didn't apply" is itself information.

7. **(Cell 12) Schema table.** Built a per-column summary (dtype, cardinality, missing+NIU rate, NIU-only rate, top value, top-value share), saved as `schema_summary.csv`. Using it as a checklist for preprocessing: columns over 50% missing+NIU are drop candidates (or need an "applicable / not applicable" flag); columns where one value covers 95%+ are near-constant and probably useless; columns with more than 50 distinct values will need target or frequency encoding rather than one-hot.

8. **(Cells 13–16) Feature-target exploration.** Looked at how each feature relates to the target. For numerics: mean/median/std by class. For low-cardinality categoricals (≤15 unique): positive rate per category. For high-cardinality ones: top categories only, with a min-count threshold so sparse cells don't mislead. Also profiled capital gains, capital losses, and dividends from stocks for zero-inflation — they're heavily zero-inflated, and within the nonzero rows the positive rate is way above baseline. I'll add a "has nonzero" flag in preprocessing; trees will probably pick this up on their own but it'll help the logistic baseline.

9. **(Cell 17) Mutual information ranking.** Ran MI against the target on a 50K-row sample. The encoding here (ordinal on categoricals, median fill on numerics) is throwaway, just for getting a ranking — won't be reused for modeling. Exported the top 25 and bottom 10 as `mi_ranking.csv`. Top ones are the shortlist; bottom ones are drop candidates I'll cross-reference with the redundancy check before pulling the trigger.


### 1_preprocessing.ipynb

1. **(Cell 3) Column Dropping Strategy.** `hispanic_origin`, `own_business_or_self_employed` carries semantic and administrative meaning, thus kept despite low MI ranks. `wage_per_hour` supposedly carries information from literal meaning, yet is inflated by 0's. Full-time employees do not report, and is also affected by hours worked per year. Low MI rank also confirmed it, thus dropped. The migration related columns are dropped, due to 1. low MI 2. large amount of missing values due to questionnaire difference between year 94 and 95 identified in eda.
2. **(Cell 6-10) Engineered features — rationale.**

- **`education_ordinal`** — Education has a clear natural ordering that one-hot encoding discards. The logistic baseline shouldn't need 17 separate coefficients to learn "more education → higher income"; an ordinal coefficient captures that monotonic trend directly. Trees are indifferent to the choice, but the linear baseline becomes meaningfully more competitive with the ordinal representation, which keeps the baseline-vs-XGBoost comparison honest.

- **`n_income_sources`** — Trees can learn an AND interaction across the three wealth indicators (earned income, capital gains, dividends) on their own; the logistic baseline can't. Without this feature, the linear model treats the indicators independently and misses the stronger signal that having multiple sources amplifies. Pre-computing the count surfaces wealth-diversity as a single dimension.

- **`is_employed`** — The "in labor force" signal is fragmented across `weeks worked in year`, `num persons worked for employer`, and several "Not in universe" markers, each carrying a piece. Surfacing it as a single binary makes the structural signal explicit for both models, simplifies SHAP plots in the report, and explains away the multicollinearity observed between `weeks worked` and `num persons worked for employer` (r=0.75, almost entirely driven by shared structural zeros).

- **`is_self_employed`** — Self-employed people have a qualitatively different income profile: lower reported wages, higher capital gains and dividends, much wider variance. Both models can find this in `class of worker` on their own, but extracting the binary makes the interaction with the wealth-indicator features cleaner and gives the report a more legible interpretation than "the coefficient on `class of worker = Self-employed-not incorporated` was X."

- **`married_joint_filer`** — The dataset has no direct spouse-income field, but two-earner households are a well-known driver of >$50K outcomes. Joint tax filing is the cleanest available proxy since it implies a married couple where both spouses have filing-relevant income. Without this feature, the model has to infer the dual-earner effect from messy combinations of marital status, household composition, and tax filer status on its own.

3. **(Cell 12) Train/test split.** Used a stratified random 80/20 split rather than a temporal split (1994 → 1995). The dataset is cross-sectional with two consecutive CPS waves under consistent methodology, so the IID assumption is reasonable — the extra statistical power from random splitting was preferred over the modest robustness signal a year-split would have given. The genuine year-related distribution shift in this dataset lived in the migration-status questions (Cramér's V = 1.0 with `year`), and those columns were already dropped prior to modeling for that reason, removing the main motivation a temporal split would have addressed. A temporal split is worth running as a robustness check in any production deployment but isn't required as the primary validation strategy for cross-sectional data like this.

4. **(Cell 14) log1p transform on zero-inflated numerics.** Capital gains, capital losses, and dividends from stocks have ~95% zero rates with heavy-tailed nonzero distributions. Raw StandardScaler on these columns squashes the zero mass into a noisy band and lets the tail dominate scaling, which obscures the structural zero-vs-nonzero signal. Applied log1p before scaling: zeros stay at zero (since log(1+0)=0), the tail compresses, and the scaled values become more linear with respect to the target. Combined with the engineered _nonzero binary flags, this gives the linear model a clean two-part representation. Trees are unaffected by the transform (monotonic).



### 2_classifier.ipynb

1. **(Cell 5) Logistic regression baseline.** Trained with class_weight='balanced' for imbalance and sample_weight=w_train for population alignment. Not tuned; the baseline exists to anchor the comparison, not to be the production model.

2. **(Cells 6–9) XGBoost and LightGBM tuning.** Here we chose gradient-boosted models, specifically for its known excellence for prediction tasks on tabular data. Both tuned via RandomizedSearchCV with 20 iterations, 3-fold stratified CV, and PR-AUC (average_precision) scoring. Search space covered tree complexity, learning rate, subsampling, and L1/L2 regularization. After tuning, models were refit on the full training data with early stopping (50 rounds) against the validation slice. CV scoring used unweighted PR-AUC for tractability; test-set evaluation retains population-weighted metric reporting.

3. **(Cell 10-11) Final model selection.** Final model selected post-hoc on population-weighted test PR-AUC. Selection on test is acceptable here because no further model changes are made after selection; the next step uses the chosen model as-is.

4. **(Cell 17) Threshold reporting.** Two operating points reported: F1-optimal threshold (best single classifier) and 80th-percentile score (top-20% marketing campaign). Default 0.5 not used since the scale_pos_weight rebalancing shifts predicted probabilities upward, making 0.5 a poor decision boundary regardless of business context.

### 3_segmentation.ipynb

1. **Segmentation Scope.** I chose to segment over the entire population over the wealthy-only group, to align with the prompt's description of "segmenting the people the dataset represents". A comprehensive understanding of the group of people can also contribute to a more personalized marketing plan targeting different customer cohorts with different products/channels. 

2. **(Cell 5) Feature Selected for Segmentation.** Seven features chosen for K-means: age, education_ordinal, weeks worked in year, is_employed, is_self_employed, married_joint_filer, is_male. Selection criteria: (1) interpretability — each feature maps cleanly to a marketing-recognizable axis (life stage, socioeconomic status, labor force activity, household structure, gender); (2) dimensionality control — K-means with Euclidean distance loses discriminative power above ~15 features, especially with mixed types; (3) complementarity with the classifier — wealth indicators (capital gains, dividends, n_income_sources) deliberately excluded so segmentation reveals demographic structure rather than rediscovering the income boundary; (4) one-hot encoding cost — high-cardinality categoricals (industry, occupation, country, race) would add 60+ binary dummies and overwhelm the continuous features in distance calculations; (5) fair-marketing considerations — race and country-of-origin excluded as primary clustering axes to avoid disparate-impact risk in premium-product targeting (these are reported as descriptive segment characteristics in the report's profile table instead).

3. **(Cell 6) Segmentation Algorithm Choice.** K-means with population-weighted centroids (sample_weight=weight). Continuous features standardized; binary features left on native 0/1 scale (already comparable magnitudes; rescaling would distort the relative weight given to demographic vs. employment dimensions).

4. **(Cell 7) Choosing best K - number of groups.** K selected for interpretability alongside elbow/silhouette. Looking at Inertia and Silloutte score, k=3 actually gives the best result. However after inspection and comparison, k=4 gives better interpretation and rationale over a distinct profile - early career/students. Final K=4 — the K that produced personas distinguishable along life-stage and employment-status axes that a marketing team can act on.

5. **Population weights used throughout.** — in K-means fitting, profile statistics, and cross-tab metrics. Segment sizes reported as population shares, not sample shares.


### 4_fairness.ipynb

1. **(Cells 1–6) Fairness Analysis.** Performed adverse-impact analysis on the classifier predictions by sex, race, and Hispanic origin, computing selection rate, disparate impact ratio, TPR, FPR, and precision by group at the F1-optimal threshold. Segment-level demographic composition also analyzed. Findings flagged in the report's fairness section. No mitigation attempted within the take-home scope; this is documented as a pre-deployment requirement rather than a present remediation.