# 💰 Financial Wellness App — Market Validation Analytics

End-to-end data-analytics project validating demand for a subscription
**financial-wellness & micro-investing app** for young earners in India.
Includes EDA, diagnostics, K-Means segmentation, five ML classifiers, and a
deployable **Streamlit dashboard**.

> Dataset is **synthetic** (1,000 respondents, 93 variables), generated to mirror
> realistic relationships for an academic analytics project.

---

## 🚀 Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. (optional) regenerate all figures, tables and trained models
python run_analysis.py

# 3. launch the dashboard
streamlit run app.py
```

### Deploy to Streamlit Community Cloud
Push this repo to GitHub → https://share.streamlit.io → New app → select the repo
and `app.py`. `requirements.txt` is detected automatically.

---

## 📁 Project structure

```
financial_wellness_app/
├── app.py                       # Streamlit dashboard (11 pages)
├── run_analysis.py              # master pipeline: EDA → diagnostics → segmentation → ML
├── requirements.txt
├── README.md
├── data/
│   └── survey_data_clean.csv    # cleaned, analysis-ready dataset
├── src/                         # reusable modules
│   ├── config.py                # paths, feature lists, theme
│   ├── data_loader.py           # data loading
│   └── modeling.py              # model zoo + tuning grids
├── models/                      # trained artifacts (.pkl) + metrics.json
│   ├── best_model.pkl           # deployed classifier (Random Forest)
│   ├── model_*.pkl              # every trained classifier
│   ├── kmeans.pkl, segment_scaler.pkl
│   ├── feature_columns.json
│   └── metrics.json             # consolidated results
├── outputs/
│   ├── figures/                 # all visualizations (.png)
│   └── tables/                  # data dictionary, diagnostics, comparisons (.csv)
└── reports/
    └── business_analytics_report.md
```

---

## 📊 Dashboard pages
Home · About · Dataset Overview · Exploratory Data Analysis · Diagnostic Analysis ·
Customer Segmentation · Machine Learning Models · Model Comparison ·
**Subscription Prediction** (live) · Business Insights · Conclusion.

Interactive filters, Plotly charts, a live subscription-probability predictor, and
CSV download buttons.

---

## 🔑 Headline findings
- **28.5%** of respondents are subscription-ready today.
- Adoption is driven by **trial likelihood, willingness to pay and adoption readiness** — not demographics.
- Three segments: **Premium-Ready Professionals** (60% conversion), **Financially Stressed Beginners**, **Smart Savers**.
- **Random Forest** predicts subscription at **84.5% accuracy / 0.85 ROC-AUC** — recommended for lead scoring.
- Recommended pricing: **freemium**, premium tier near **₹149/month**.

See `reports/business_analytics_report.md` for the full write-up.

---

## 🧪 Methods
scikit-learn (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting),
K-Means with Elbow/Silhouette, SciPy tests (chi-square, Welch t-test, ANOVA, Spearman)
with Cramér's V and Cohen's d effect sizes.

## ♻️ Reproducibility
`python run_analysis.py` regenerates every artifact from `data/survey_data_clean.csv`.
Synthetic data can be rebuilt with the generator from earlier project phases.
