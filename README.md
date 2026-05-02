
 # Santander Bank Loan Default Prediction — Decision Tree & Random Forest

> *A supervised classification study comparing Decision Tree and Random Forest models for small business loan default prediction at Santander Bank. GridSearchCV with 5-fold cross-validation optimises both models. The Decision Tree is recommended for deployment on the basis of its superior default recall (0.87), interpretability, and regulatory compliance — despite the Random Forest achieving higher overall accuracy.*

---

## Table of Contents
1. [Business Problem](#1-business-problem)
2. [Dataset](#2-dataset)
3. [Descriptive Statistics](#3-descriptive-statistics)
4. [Analytical Pipeline](#4-analytical-pipeline)
5. [Figures and Business Findings](#5-figures-and-business-findings)
6. [Model Results Tables](#6-model-results-tables)
7. [Recommendation](#7-recommendation)
8. [Bugs Fixed](#8-bugs-fixed)
9. [Tech Stack](#9-tech-stack)
10. [How to Run](#10-how-to-run)

---

## 1. Business Problem

Santander Bank operates a portfolio of small business loans in the US market. The challenge is identifying which loans are at risk of default **before** the default occurs. Proactive identification enables early action — restructuring repayment terms, increasing monitoring, or adjusting credit limits — all of which improve recovery outcomes.

The classification task is binary: predict whether a loan status will be **Current** (performing normally) or **Default** (in non-payment). The two error types carry critically asymmetric costs:

| Error Type | Description | Cost |
|---|---|---|
| **False Negative** | Actual default missed by the model | Full outstanding loan balance lost — operationally catastrophic |
| **False Positive** | Current loan flagged as potential default | One unnecessary review call — trivial |

This asymmetry makes **recall for the default class** the primary evaluation metric — not accuracy.

---

## 2. Dataset

**Source:** `LoanDefault.csv` (risperaketch repository, GitHub)
**Size:** 5,261 loans · 6 variables · **0 missing values**

| Variable | Type | Encoding in Pipeline | Description |
|---|---|---|---|
| `Status` | Binary Target | Current→0, Default→1 | Loan performance status |
| `Credit.Grade` | Ordinal | `.codes` → int8 (AA=0…NC=7) then StandardScaler | 8-level ordered credit rating |
| `Amount` | Numeric | SimpleImputer → StandardScaler | Original loan amount ($) |
| `Age` | Numeric | SimpleImputer → StandardScaler | Age of the loan in years |
| `Borrower.Rate` | Numeric | SimpleImputer → StandardScaler | Loan interest rate |
| `Debt.To.Income.Ratio` | Numeric | SimpleImputer → StandardScaler | Borrower debt-to-income ratio |

**Class distribution:**

| Class | Label | Count | Proportion |
|---|---|---|---|
| 0 | Current | 5,186 | **98.57%** |
| 1 | Default | 75 | **1.43%** |
| — | Imbalance ratio | — | **~69 : 1** |

---

## 3. Descriptive Statistics

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| Amount ($) | 4,852.77 | 4,466.05 | 1,000 | 3,001 | 35,000 |
| Age (years) | 4.40 | 2.74 | 0 | 4 | 14 |
| Borrower.Rate | 0.1904 | 0.0681 | 0.000 | 0.190 | 0.365 |
| Debt.To.Income.Ratio | 48.39 | 1,069.29 | 0.000 | 0.160 | 10.01 |
| Credit.Grade (encoded 0–7) | 3.70 | 1.91 | 0 | 4 | 7 |

**Key observations:**
- `Debt.To.Income.Ratio` has extreme skew — std = 1,069 driven by a small number of outliers; median 0.16 is far below the mean 48.39
- `Amount` is dominated by micro-loans — 75% of loans are under $6,000
- `Borrower.Rate` clusters between 14% and 23% for most loans; rates above 25% signal pre-assessed high risk
- Mean `Age` of 4.4 years reflects a portfolio still predominantly in early repayment stages

---

## 4. Analytical Pipeline

```
LoanDefault.csv  (5,261 rows × 6 columns, 0 missing)
       ↓
Q0 — Load · Info · Type identification · Target encoding · EDA
       Status → binary int 0/1
       Credit.Grade → ordinal int8 via .codes (AA=0 … NC=7)
       Age → numeric (corrected from nominal classification)
       ↓
Q1 — Feature / Target split + 80/20 stratified split
       Train: 4,208 rows | Test: 1,053 rows | stratify=y
       ↓
Q2 — Preprocessing Pipeline (ColumnTransformer)
       All 5 features → SimpleImputer(median) → StandardScaler
       (single transformer — Credit.Grade already int, no OrdinalEncoder needed)
       ↓
Q3 — Decision Tree GridSearchCV (5-fold, accuracy scoring)
       Pipeline: preprocessor → DecisionTreeClassifier(class_weight='balanced')
       Grid: max_depth=[2,3,4,5] × min_samples_split=[2..10] → 180 model fits
       Result: max_depth=5, min_samples_split=2, CV accuracy=0.8703
       ↓
Q4 — Feature importance + Decision Tree visualization
       Most important: Age (0.8437)
       ↓
Q5 — Random Forest GridSearchCV (5-fold, accuracy scoring)
       Pipeline: preprocessor → RandomForestClassifier(class_weight='balanced')
       Grid: n_estimators=[50,100,150,200,250] → 25 model fits
       Result: n_estimators=150, CV accuracy=0.9867
       ↓
Q6 — Test predictions · Classification reports · Confusion matrices
       ROC curves · Model comparison summary · Recommendation
```

---

## 5. Figures and Business Findings

---

### Figure 1 — Target Variable Distribution

<img width="1199" height="557" alt="image" src="https://github.com/user-attachments/assets/79b829cf-ca34-4e3a-b7d4-fdb002d6515f" />


**What it shows:**
Two panels presenting the binary class distribution of the `Status` target. Left: a bar chart with counts and percentages annotated above each bar. Right: a pie chart showing proportional breakdown. Blue = Current (0); Red = Default (1). The chart title states the imbalance ratio explicitly.

**Exact values:**
- Current (0): **5,186 loans (98.57%)**
- Default (1): **75 loans (1.43%)**
- Imbalance ratio: **~69:1**

**Business finding:**
The 69:1 class imbalance is the most consequential dataset characteristic, driving every downstream modelling decision. A naïve classifier predicting "Current" for every loan achieves 98.57% accuracy while catching **zero** defaults — the textbook accuracy paradox in imbalanced classification. Three interventions are applied: `class_weight='balanced'` in both models (multiplies the misclassification penalty for defaults by ~69), `stratify=y` in the train/test split, and recall as the primary evaluation metric rather than accuracy. The 75 defaults in the dataset, at a mean balance of $4,853, represent over **$364,000 in at-risk capital** — establishing the financial stakes of the prediction task.

---

### Figure 2 — Feature Distributions by Loan Status

<img width="1632" height="1002" alt="image" src="https://github.com/user-attachments/assets/741e7880-fb85-4dd9-b5ef-dfba4f0cbe35" />


**What it shows:**
A 2×3 grid of overlapping histograms — one panel per predictor. Blue bars represent current loans (class 0); red bars represent defaulted loans (class 1). The degree of overlap between the two distributions indicates each feature's ability to discriminate between classes when considered in isolation.

**Panel-by-panel analysis and business findings:**

| Feature | Visual Separation | Business Interpretation |
|---|---|---|
| `Amount` | Weak | Defaults concentrate slightly in lower amounts ($1,000–$5,000); small loans may have weaker collateral. Not a strong standalone predictor |
| `Age` | Moderate | Defaults skew toward younger loan ages (years 0–4). Early-stage loans have built little equity, making them more vulnerable to borrower cash flow shocks |
| `Borrower.Rate` | Moderate-Strong | Default distribution shifts rightward. Lenders price assessed default risk into rates; a rate above 25% signals that underwriters already identified elevated risk at origination |
| `Debt.To.Income.Ratio` | Weak | Both classes concentrate near zero with similar outlier distributions; DTI alone provides limited discriminatory power in this dataset |
| `Credit.Grade` (int) | Moderate | Default loans lean toward higher integer codes (D=3, E=4, HR=5, NC=6) — worse credit ratings are associated with default. The ordinal encoding preserves this ranking |

**Overarching finding:** No single feature cleanly separates defaults from current loans — this is expected in credit risk and motivates the use of tree-based models that exploit non-linear feature combinations. The actionable insight for Santander is that loans with `Borrower.Rate > 0.25` and credit grades of HR or NC warrant enhanced monitoring independent of any model output, as these features together identify a disproportionate share of the 75 observed defaults.

---

### Figure 3 — Decision Tree Grid Search Heatmap

<img width="1207" height="422" alt="image" src="https://github.com/user-attachments/assets/d07c2c8d-07ea-4324-8de4-7592a403e6d1" />


**What it shows:**
A colour-coded heatmap of 5-fold cross-validated accuracy scores for all 36 combinations of `max_depth` (rows: 2–5) and `min_samples_split` (columns: 2–10). Each cell is annotated with its exact CV accuracy. Darker orange-red = higher accuracy. Total model fits: 180.

**Key results:**

| Configuration | CV Accuracy |
|---|---|
| **max_depth=5, min_samples_split=2** | **0.8703 (optimal)** |
| max_depth=4, min_samples_split=2 | slightly lower |
| max_depth=2, min_samples_split=2 | lowest |

**Business finding:**
Reading down the rows, accuracy increases monotonically with `max_depth` — deeper trees capture more complex risk interactions. Reading across columns, `min_samples_split` produces negligible variation at any given depth — tree depth is the dominant tuning dimension. This tells Santander's analytics team that the loan default problem is moderately complex: a depth-5 tree (up to 32 leaf nodes) is required to adequately partition the risk space across all 8 credit grades, multiple loan age brackets, and the range of interest rates. Shallower trees (depth 2–3) consistently underfit, meaning simple two-to-three-rule policies cannot capture the default risk structure in this dataset.

---

### Figure 4 — Decision Tree Feature Importance

<img width="972" height="532" alt="image" src="https://github.com/user-attachments/assets/16803906-9713-417d-b470-60f1b4aaefec" />


**What it shows:**
A horizontal bar chart ranking all 5 features by their total Gini impurity reduction contribution across all Decision Tree splits. Features are sorted highest to lowest importance. The most important feature is highlighted in red; all others in blue. Each bar is annotated with its exact importance value.

**Exact importance scores:**

| Rank | Feature | Importance | % of Total |
|---|---|---|---|
| 1 | `Age` | **0.8437** | **84.4%** |
| 2 | `Borrower.Rate` | 0.0879 | 8.8% |
| 3 | `Credit.Grade` | 0.0436 | 4.4% |
| 4 | `Debt.To.Income.Ratio` | 0.0247 | 2.5% |
| 5 | `Amount` | 0.0000 | 0.0% |

**Business finding:**
`Age` (loan tenure) dominates with 84.4% of all predictive information — a finding that initially appears counter-intuitive but reflects a well-documented credit risk dynamic. Default probability is not uniformly distributed across a loan's life. In early years (0–3), borrowers have established minimal repayment history and remain fully exposed to business disruptions. As loans mature, surviving borrowers are a self-selected cohort of higher-quality creditors — the weakest have defaulted or prepaid. `Amount` contributes zero importance, indicating that loan size alone adds no discriminatory power once age, rate, grade, and DTI are known. **Operational implication for Santander:** the highest-intensity monitoring resources should be concentrated on loans in years 0–3, where the model confirms default risk is highest.

---

### Figure 5 — Decision Tree Structure Visualization

<img width="2182" height="862" alt="image" src="https://github.com/user-attachments/assets/401529f5-f0c8-4094-bd3a-011ef8747bd6" />


**What it shows:**
A graphical rendering of the top 3 levels of the optimally tuned Decision Tree (max_depth=5, min_samples_split=2). Each node displays: splitting feature and threshold, Gini impurity, sample count, class value distribution [Current, Default], and majority-class prediction. Blue nodes favour "Current"; orange nodes favour "Default". Displayed depth is capped at 3 for readability (full tree has up to 5 levels).

**Structure interpretation:**
- **Root node (level 0):** First split on `Age` — confirming it as the primary discriminator. This single threshold divides the 4,208 training samples into two subpopulations with markedly different default concentrations
- **Level 1 nodes:** Both post-`Age` branches use `Borrower.Rate` as the secondary discriminator — high-rate loans are separated within each age cohort, providing a second risk filter
- **Level 2 nodes:** `Credit.Grade` and `Debt.To.Income.Ratio` appear as tertiary splits, refining predictions within age-and-rate stratified groups

**Business finding:**
The tree structure translates directly into plain-language credit monitoring rules that Santander's relationship managers can apply without a statistical model. A representative rule might read: *"Flag for enhanced monitoring if loan age < 3 years AND Borrower.Rate > 22% — proceed to credit grade check."* This is precisely the explainability required under ECOA and Fair Lending regulations. When a customer's loan terms are modified based on model output, the bank must provide an adverse action notice in plain language. The Decision Tree's rule set is natively compliant; a Random Forest of 150 trees is not.

---

### Figure 6 — Feature Importance Comparison: Decision Tree vs. Random Forest

<img width="1632" height="563" alt="image" src="https://github.com/user-attachments/assets/9e4f4917-8d84-4f95-957b-3ee40fbeb60c" />


**What it shows:**
Two side-by-side horizontal bar charts. Left panel (blue): Decision Tree feature importances at max_depth=5. Right panel (green): Random Forest feature importances at n_estimators=150. Feature bars are sorted within each panel by importance descending.

**Analytical observation:**
The Decision Tree concentrates 84.4% of its importance in `Age` — a consequence of its greedy single-tree structure that selects the globally optimal split at each node. Once `Age` claims the root split, downstream nodes have limited residual information to attribute to other features. The Random Forest, by contrast, builds 150 trees each on a bootstrap sample with random feature subsets, preventing `Age` from dominating every tree. The RF panel shows a more balanced distribution across all five features, better reflecting each feature's marginal contribution when others are held constant.

**Business finding:**
The divergence between panels reveals that `Borrower.Rate`, `Credit.Grade`, and `Debt.To.Income.Ratio` carry more independent predictive information than the single Decision Tree's structure suggests. They are partially masked by `Age`'s dominant early splits. From a credit policy standpoint, this means Santander should monitor **all four** of these dimensions — not just loan age — in its early-warning system. A bank that reads only the Decision Tree's importance chart and ignores Borrower.Rate and Credit.Grade in its monitoring framework would leave significant risk-detection capability unused.

---

### Figure 7 — Confusion Matrices

<img width="1178" height="563" alt="image" src="https://github.com/user-attachments/assets/d9504548-dc38-4267-9385-fd39429f1a84" />

**What it shows:**
Two 2×2 confusion matrices side by side on the test set (n=1,053). Columns represent predicted classes; rows represent actual classes. Title above each matrix shows overall accuracy and default recall.

```
                    Predicted
                Current  |  Default
Actual Current  [  TN   |    FP  ]   ← false alarms (costly in time)
Actual Default  [  FN   |    TP  ]   ← missed defaults (costly in dollars)
```

**Estimated cell values from model metrics:**

| | Decision Tree | Random Forest |
|---|---|---|
| TN (correct current) | ~871 | ~1,035 |
| FP (false alarms) | ~167 | ~3 |
| FN (missed defaults) | **~2** | **~12** |
| TP (caught defaults) | **~13** | **~3** |
| Default Recall | **0.867** | 0.200 |
| Accuracy | 0.845 | 0.987 |

*(DT predicted 174 defaults total; RF predicted 5; 15 actual defaults in test set)*

**Business finding:**
At an average loan balance of $4,853, the Random Forest's ~12 missed defaults represent approximately **$58,236 in unrecovered capital per evaluation cycle.** The Decision Tree's ~167 false alarms, at an estimated 15 minutes of analyst time each, represent approximately **42 hours of review work** — an operational cost the bank can absorb. The Decision Tree catches ~13 defaults with potential recovery value of approximately **$63,089.** The financial arithmetic strongly favours accepting the false alarm burden in exchange for the higher default recovery rate.

---

### Figure 8 — ROC Curves

<img width="972" height="642" alt="image" src="https://github.com/user-attachments/assets/184953b2-5a93-4957-9671-72f14b96ba66" />


**What it shows:**
ROC curves for both models plotted on the same axes. True Positive Rate (recall) on the y-axis; False Positive Rate (1−specificity) on the x-axis. Each curve spans all decision probability thresholds. The diagonal dashed line is the random classifier baseline (AUC=0.500). AUC values are annotated in the legend.

| Model | AUC |
|---|---|
| Decision Tree | 0.8916 |
| Random Forest | **0.9082** |
| Random baseline | 0.500 |

**Business finding:**
Both models substantially outperform random prediction, confirming genuine predictive signal in the five features. The Random Forest's AUC of 0.9082 is marginally higher — its probability scores have slightly better calibration for ranking loans by default risk. This suggests that if Santander pursues **threshold recalibration** as a next step (lowering the decision boundary below 0.50 to catch more defaults at a tolerable false alarm rate), the Random Forest's better-calibrated probabilities would be the preferred starting point. Each point on the Decision Tree's ROC curve at (FPR=0.16, TPR=0.87) represents its current operating configuration — achieving 87% recall while generating ~16% false positive rate. Santander can slide left along this curve by raising the threshold to reduce false alarms if the analyst review burden becomes operationally unsustainable.

---

### Figure 9 — Model Performance Comparison

<img width="1962" height="563" alt="image" src="https://github.com/user-attachments/assets/57bc2537-e10e-4c11-a547-6bcc0df2e675" />


**What it shows:**
Five side-by-side bar charts comparing both models across: Accuracy, Precision (Default), Recall (Default), F1 (Default), and AUC. Blue = Decision Tree; Green = Random Forest. Values annotated above each bar.

**Full numerical summary:**

| Metric | Decision Tree | Random Forest | Business-Preferred Winner |
|---|---|---|---|
| Accuracy | 0.8452 | **0.9867** | RF (but misleading due to imbalance) |
| Precision (Default) | 0.0747 | **0.6000** | RF |
| **Recall (Default)** | **0.8667** | 0.2000 | **DT — the critical metric** |
| F1 (Default) | 0.1376 | 0.3000 | RF (but not the right objective) |
| AUC | 0.8916 | **0.9082** | RF (marginally) |

**Business finding:**
The five-panel comparison makes the accuracy paradox explicit for any stakeholder who might be tempted to select the higher-accuracy model. The Random Forest wins on 4 of 5 displayed metrics — yet it is the wrong model for the business problem. The single metric on which the Decision Tree excels — Recall (Default) = 0.87 — directly determines how many actual defaults the bank can intervene on before losses are realised. A 20% recall (RF) means the bank acts on 3 of every 15 detected defaults per cycle; an 87% recall (DT) means it acts on 13 of 15. The Decision Tree's F1 of 0.14 — lower than RF's 0.30 — reflects its deliberate sacrifice of precision to maximise recall. With a 69:1 imbalance, a model correctly optimised for recall will generate many false positives, producing low precision and moderate F1. This is the correct engineering trade-off for a bank whose primary concern is catching defaults before they become write-offs.

---

## 6. Model Results Tables

### Decision Tree — Hyperparameter Grid Results Summary

| max_depth | min_samples_split | CV Accuracy | Selected? |
|---|---|---|---|
| 5 | **2** | **0.8703** | **YES — optimal** |
| 5 | 3–10 | slightly lower | No |
| 4 | 2 | lower | No |
| 3 | 2 | lower | No |
| 2 | 2 | lowest | No |

**Final DT configuration:** max_depth=5, min_samples_split=2
- Best CV accuracy: **0.8703**
- Training accuracy: **0.8489**
- Total grid fits: 180 (4 depths × 9 split values × 5 folds)

### Random Forest — CV Accuracy by n_estimators

| n_estimators | CV Accuracy | Std | Selected? |
|---|---|---|---|
| 50 | 0.9860 | ±0.0017 | No |
| 100 | 0.9862 | ±0.0018 | No |
| **150** | **0.9867** | **±0.0016** | **YES — optimal** |
| 200 | 0.9865 | ±0.0012 | No |
| 250 | 0.9865 | ±0.0012 | No |

**Final RF configuration:** n_estimators=150
- Best CV accuracy: **0.9867**
- Training accuracy: **0.9998** *(mild overfitting — 1.3pp gap vs. CV)*
- Total grid fits: 25 (5 values × 5 folds)

The plateau above n=150 confirms diminishing returns — performance differences above 150 trees are within one standard deviation and provide no statistical benefit.

### Full Classification Reports

**Decision Tree (Test Set, n=1,053)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Current (0) | 1.00 | 0.84 | 0.91 | 1,038 |
| **Default (1)** | 0.07 | **0.87** | 0.14 | 15 |
| Accuracy | — | — | **0.85** | 1,053 |
| Macro avg | 0.54 | 0.86 | 0.53 | 1,053 |

**Random Forest (Test Set, n=1,053)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Current (0) | ~0.99 | ~1.00 | ~0.99 | 1,038 |
| **Default (1)** | **0.60** | 0.20 | 0.30 | 15 |
| Accuracy | — | — | **0.99** | 1,053 |
| Macro avg | ~0.79 | ~0.60 | ~0.64 | 1,053 |

---

## 7. Recommendation

**Deploy: Decision Tree (max_depth=5, min_samples_split=2)**

The case rests on three pillars:

**Pillar 1 — Recall (The Primary Business Metric)**
The Decision Tree identifies **87% of actual defaults** (approximately 13 of 15 in the test set). The Random Forest identifies only **20%** (approximately 3 of 15). For a bank whose objective is catching defaults before they become unrecoverable losses, a model missing 80% of defaults provides essentially no early-warning capability. The extreme asymmetry between the cost of a false negative (full loan loss ≈ $4,853 average) and a false positive (one review call ≈ 15 analyst minutes) makes recall maximisation the correct business objective.

**Pillar 2 — Interpretability (Regulatory Requirement)**
Credit risk models used for adverse action decisions at US financial institutions are subject to ECOA, Regulation B, and Fair Lending requirements. When loan terms are modified based on model output, the bank must issue a plain-language adverse action notice explaining the decision. The Decision Tree produces an auditable rule set that compliance officers can inspect and explain to regulators without additional tooling. A 150-tree Random Forest requires post-hoc approximation (SHAP values, LIME) which regulators do not accept as primary explanations.

**Pillar 3 — Financial Impact**
In the test set, the Decision Tree catches approximately 13 of 15 defaults vs. the Random Forest's 3. At a mean balance of $4,853, this represents an additional **$48,530 in recoverable or restructurable loan value** per evaluation cycle — far exceeding the operational cost of the additional false alarms generated.

**Recommended improvements before production deployment:**
1. **Threshold calibration** — lower decision threshold from 0.50 toward 0.30 using the ROC curve to further increase recall at a controlled false-positive rate
2. **SMOTE oversampling** — synthetically augment the 75 training defaults to improve the model's representation of rare default patterns
3. **Quantitative cost-sensitive weights** — compute `class_weight` analytically from the actual dollar loss per false negative rather than using the generic `'balanced'` setting
4. **Temporal hold-out validation** — validate on loans originated in a later period to test deployment robustness across different economic conditions
5. **Quarterly model retraining** — default rates and feature correlates shift with macroeconomic cycles; rolling retraining maintains model currency

---

## 9. Tech Stack

```
Python 3.10
├── pandas          — data loading, descriptive statistics, summary tables
├── NumPy           — array operations, random seed
├── scikit-learn    — Pipeline, ColumnTransformer, GridSearchCV (5-fold)
│                     DecisionTreeClassifier, RandomForestClassifier
│                     SimpleImputer(median), StandardScaler
│                     classification_report, confusion_matrix
│                     ConfusionMatrixDisplay, roc_auc_score, roc_curve
│                     precision_score, recall_score, f1_score, clone
├── matplotlib      — all charts, plot_tree visualization
└── seaborn         — heatmap, distribution histograms, styled themes
```

---

## 10. How to Run

**Google Colab (recommended)**
```python
# Runtime → Run all
# Dataset= upload loan defaul.csv data into the colab then run all


**Local Jupyter**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Santander Bank Loan Default Prediction.ipynb
```

---

## Skills Demonstrated

`Decision Tree` · `Random Forest` · `Ensemble Methods` · `Hyperparameter Tuning` · `GridSearchCV` · `5-Fold Cross-Validation` · `Class Imbalance` · `class_weight=balanced` · `sklearn Pipeline` · `ColumnTransformer` · `StandardScaler` · `Recall Optimisation` · `ROC-AUC` · `Confusion Matrix` · `Gini Feature Importance` · `Credit Risk Analytics` · `Regulatory Interpretability (ECOA)` · `Python` · `scikit-learn`

---

## Author

**Aketch Adhiambo Okoth**
M.S. Business Analytics — Montclair State University 
[LinkedIn](https://linkedin.com/in/your-profile) · [Portfolio](https://your-portfolio-url.com)
"""

