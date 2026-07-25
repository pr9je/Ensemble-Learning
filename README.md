# 🚗 Ola Driver Attrition Prediction — Ensemble Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![Domain](https://img.shields.io/badge/Domain-HR%20Analytics-purple?style=flat)

> **Can we predict which Ola drivers are likely to leave — before they do?**  
> A complete ML pipeline comparing 5 ensemble methods to identify the strongest predictors of driver attrition and deliver actionable retention recommendations.

---

## 📌 Problem Statement

Ola, one of India's largest ride-hailing platforms, faces significant operational costs when drivers churn — recruitment, onboarding, and lost rides. This project builds a **binary classification model** to predict driver attrition using historical HR data, enabling Ola's operations team to intervene proactively with at-risk drivers.

**Business Question:** *Which drivers are most likely to leave Ola in the next quarter — and why?*

---

## 📊 Key Findings

| Finding | Insight |
|---|---|
| 🏆 Best model | XGBoost — AUC 0.91 |
| 📉 Top predictor | Quarterly rating drop (most important feature) |
| 💰 2nd predictor | Income level — lower income = higher attrition |
| 📅 3rd predictor | Tenure — first 6 months are highest risk window |
| ⚠️ False negative cost | Missing an at-risk driver costs more than a false alarm |

**Recommendation:** Ola should prioritise drivers with a rating drop of ≥ 0.5 points quarter-on-quarter AND tenure under 6 months for proactive retention interventions (e.g. bonuses, support calls).

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| Source | Provided as part of Scaler MSc AI & ML curriculum |
| Rows | ~3,000+ driver records |
| Target variable | `Target` — 1 = churned, 0 = retained |
| Features | Demographics, income, quarterly ratings, joining date, city, education |
| Class imbalance | Yes — handled via class weighting and threshold tuning |

**Key features used:**
- `quarterly_rating` — current and previous quarter rating
- `income` — monthly income bracket
- `joining_designation` — seniority at time of joining
- `total_business_value` — revenue generated
- `grade` — performance grade
- `age`, `gender`, `education_level`

---

## 🔬 Methodology

```
Raw Data
   │
   ▼
Exploratory Data Analysis (EDA)
   │  ├── Missing value analysis
   │  ├── Distribution plots
   │  ├── Correlation heatmap
   │  └── Target class imbalance check
   │
   ▼
Feature Engineering
   │  ├── Quarterly rating delta (current vs previous)
   │  ├── Tenure calculation from joining date
   │  └── Encoding categorical variables
   │
   ▼
Model Training & Comparison
   │  ├── Random Forest (Bagging)
   │  ├── AdaBoost (Boosting)
   │  ├── Gradient Boosting (Boosting)
   │  ├── XGBoost (Boosting)
   │  └── Stacking Classifier
   │
   ▼
Evaluation
   │  ├── ROC-AUC score
   │  ├── Confusion matrix
   │  ├── Precision-Recall curve
   │  └── Threshold tuning (minimise false negatives)
   │
   ▼
Feature Importance (SHAP)
   └── Business recommendations
```
---

## 🛠️ Tech Stack

```python
# Core
pandas==2.0        # Data manipulation
numpy==1.24        # Numerical operations
matplotlib==3.7    # Visualisation
seaborn==0.12      # Statistical plots

# Machine Learning
scikit-learn==1.3  # RF, AdaBoost, GradientBoosting, Stacking
xgboost==1.7       # XGBoost classifier
shap==0.42         # Feature importance explainability

# Evaluation
scipy==1.10        # Statistical tests
```
---

## 💡 Business Recommendations

Based on the model outputs and SHAP analysis:

1. **Target high-risk segment proactively** — drivers with a quarterly rating drop ≥ 0.5 AND tenure < 6 months represent the highest-risk cohort. Assign a relationship manager to this group.

2. **Income intervention** — the model identifies income as the second strongest predictor. A performance-linked bonus structure for drivers in the bottom income quartile could reduce early churn by an estimated 15–20%.

3. **Onboarding programme** — since tenure < 6 months is a strong risk factor, a structured 3-month onboarding experience (mentorship, guaranteed minimum earnings) could significantly improve early retention.

4. **Early warning dashboard** — deploy the XGBoost model as a monthly scoring pipeline to flag at-risk drivers before they resign, giving the operations team a 4–6 week intervention window.

---

## 🔮 Future Work

- [ ] Deploy model as a REST API using FastAPI
- [ ] Build a Streamlit dashboard for real-time driver risk scoring
- [ ] Integrate SMOTE for class imbalance instead of class weighting
- [ ] Test LightGBM and CatBoost as additional ensemble methods
- [ ] Add time-series component — predict attrition within specific time window

---

## 👤 Author

**Mitesh Prajapati**  
MSc in Artificial Intelligence & Machine Learning — Woolf University  
Ex-Associate Engineer R&D, Blue Star Ltd.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-pmitesh-blue?style=flat&logo=linkedin)](https://linkedin.com/in/pmitesh)
[![GitHub](https://img.shields.io/badge/GitHub-pr9je-black?style=flat&logo=github)](https://github.com/pr9je)
[![Portfolio](https://img.shields.io/badge/Portfolio-pr9je.github.io-green?style=flat)](https://pr9je.github.io)
[![Email](https://img.shields.io/badge/Email-miteshprajapati936%40gmail.com-red?style=flat&logo=gmail)](mailto:miteshprajapati936@gmail.com)

---

*⭐ If you found this project useful, consider giving it a star — it helps others discover the work.*
