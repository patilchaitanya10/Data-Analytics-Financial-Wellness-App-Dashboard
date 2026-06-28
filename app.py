"""
============================================================================
Financial Wellness App - Market Validation Dashboard (Streamlit)
Run locally:  streamlit run app.py
Deploy:       push to GitHub -> Streamlit Community Cloud -> app.py
============================================================================
"""
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Financial Wellness App - Analytics",
                   page_icon="💰", layout="wide", initial_sidebar_state="expanded")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "survey_data_clean.csv")
MODELS = os.path.join(ROOT, "models")
FIGS = os.path.join(ROOT, "outputs", "figures")
TBLS = os.path.join(ROOT, "outputs", "tables")
PRI, ACC = "#1F3864", "#2E75B6"

# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA):
        st.error(
            "❌ **Data file not found:** `data/survey_data_clean.csv`.\n\n"
            "The app loaded but the dataset folder is missing from the deployment. "
            "On Streamlit Cloud this almost always means the `data/`, `models/`, and "
            "`outputs/` folders were not committed to your GitHub repository.\n\n"
            "**Fix:** make sure these folders are pushed to the repo (see README), "
            "then reboot the app from *Manage app → Reboot*."
        )
        st.caption(f"Looked in: {DATA}")
        st.stop()
    df = pd.read_csv(DATA)
    ca = os.path.join(TBLS, "cluster_assignments.csv")
    if os.path.exists(ca):
        df = df.merge(pd.read_csv(ca), on="RespondentID", how="left")
    return df

@st.cache_data
def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}

@st.cache_data
def load_table(name):
    p = os.path.join(TBLS, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_resource
def load_model(name):
    p = os.path.join(MODELS, name)
    return joblib.load(p) if os.path.exists(p) else None

df = load_data()
metrics = load_json(os.path.join(MODELS, "metrics.json"))
features = load_json(os.path.join(MODELS, "feature_columns.json"))

def fig_path(n):
    return os.path.join(FIGS, n)
def show_fig(n, caption=None):
    p = fig_path(n)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.info(f"Figure '{n}' not found — run run_analysis.py to generate it.")
def dl_button(frame, label, fname):
    st.download_button(label, frame.to_csv(index=False).encode(), fname, "text/csv")

# --------------------------------------------------------------------------
st.sidebar.title("💰 Financial Wellness App")
st.sidebar.caption("Market Validation Analytics")
PAGES = ["Home", "About the Project", "Dataset Overview", "Exploratory Data Analysis",
         "Diagnostic Analysis", "Customer Segmentation", "Machine Learning Models",
         "Model Comparison", "Subscription Prediction", "Business Insights", "Conclusion"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.metric("Respondents", f"{len(df):,}")
st.sidebar.metric("Subscribe rate", f"{df['Subscribe_Binary'].mean()*100:.1f}%")
if metrics.get("modeling"):
    st.sidebar.metric("Best model", metrics["modeling"]["best_model"])

# ==========================================================================
if page == "Home":
    st.title("Financial Wellness & Micro-Investing App")
    st.subheader("Market Validation Dashboard — Young Earners, India")
    st.markdown("""
This interactive dashboard validates demand for a **subscription financial-wellness app**
using a survey of 1,000 young Indian earners. Explore who is ready to subscribe, what
drives adoption, how customers segment, and predict subscription intent live.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{len(df):,}")
    c2.metric("Subscribe-ready", f"{df['Subscribe_Binary'].mean()*100:.1f}%")
    c3.metric("Willing to pay", f"{df['WTP_Positive'].mean()*100:.1f}%")
    if metrics.get("modeling"):
        best = pd.DataFrame(metrics["modeling"]["comparison"]).iloc[0]
        c4.metric("Best model ROC-AUC", f"{best['roc_auc']:.2f}")
    st.markdown("---")
    cc = df["Q25_Subscribe"].value_counts().reindex(["Yes","Maybe","No"])
    fig = px.pie(values=cc.values, names=cc.index, hole=.45,
                 color=cc.index, color_discrete_map={"Yes":ACC,"Maybe":"#9DC3E6","No":"#D6DCE5"},
                 title="Subscription Intent (Q25)")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Use the sidebar to explore each stage of the analysis.")

# ==========================================================================
elif page == "About the Project":
    st.title("About the Project")
    st.markdown("""
### Objective
Validate the commercial viability of a subscription **financial-wellness and micro-investing app**
targeting young earners (ages ~18–35) in India, and identify the highest-potential customers,
features, and pricing.

### Business questions
1. What share of the market is ready to subscribe today?
2. Which behaviours and attitudes drive subscription intent?
3. How do customers segment, and how should each segment be served?
4. Can we predict subscription intent to prioritise marketing spend?

### Approach
Survey design → synthetic data generation → cleaning & feature engineering →
descriptive & diagnostic analytics → K-Means segmentation → predictive modelling →
business recommendations → this dashboard.

### Methods
Descriptive statistics & visual EDA · Chi-square, t-tests, ANOVA, correlation & effect sizes ·
K-Means with Elbow/Silhouette · Logistic Regression, KNN, Decision Tree, Random Forest,
Gradient Boosting with cross-validated tuning.

*The dataset is synthetic, built to reflect realistic relationships for an academic analytics project.*
""")

# ==========================================================================
elif page == "Dataset Overview":
    st.title("Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Observations", f"{df.shape[0]:,}")
    c2.metric("Variables", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))
    st.markdown("### Data Dictionary")
    dd = load_table("data_dictionary.csv")
    if not dd.empty:
        roles = ["All"] + sorted(dd["role"].unique().tolist())
        pick = st.selectbox("Filter by role", roles)
        view = dd if pick=="All" else dd[dd["role"]==pick]
        st.dataframe(view, use_container_width=True, height=360)
        dl_button(dd, "⬇ Download data dictionary", "data_dictionary.csv")
    st.markdown("### Summary Statistics (numeric)")
    ss = load_table("summary_statistics.csv")
    st.dataframe(ss, use_container_width=True, height=300)
    st.markdown("### Sample of the data")
    st.dataframe(df.head(20), use_container_width=True)
    dl_button(df, "⬇ Download full dataset", "survey_data_clean.csv")

# ==========================================================================
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("Interactively explore distributions and relationships. Use the filters to subset respondents.")
    with st.expander("🔎 Filters", expanded=True):
        c1,c2,c3 = st.columns(3)
        emp = c1.multiselect("Employment", sorted(df["Q5_Employment"].unique()))
        tier = c2.multiselect("City tier", sorted(df["Q3_CityTier"].unique()))
        age = c3.slider("Age range", int(df["Q1_Age"].min()), int(df["Q1_Age"].max()),
                        (int(df["Q1_Age"].min()), int(df["Q1_Age"].max())))
    d = df.copy()
    if emp: d = d[d["Q5_Employment"].isin(emp)]
    if tier: d = d[d["Q3_CityTier"].isin(tier)]
    d = d[(d["Q1_Age"]>=age[0]) & (d["Q1_Age"]<=age[1])]
    st.caption(f"{len(d)} respondents in current selection")

    t1,t2,t3,t4 = st.tabs(["Distributions","By Subscribe","Correlation","Cross-tabs"])
    with t1:
        col = st.selectbox("Variable", ["Q1_Age","IncomeMidpoint_INR","Q22_WTP_INR",
              "AdoptionReadiness","FinDiscipline_Index","PainPoint_Index","Q24_TrialLikelihood"])
        st.plotly_chart(px.histogram(d, x=col, nbins=30, marginal="box",
                        color_discrete_sequence=[ACC]), use_container_width=True)
    with t2:
        col = st.selectbox("Numeric variable", ["AdoptionReadiness","FinDiscipline_Index",
              "PainPoint_Index","Q22_WTP_INR","Q24_TrialLikelihood","IncomeMidpoint_INR"], key="bs")
        d2 = d.copy(); d2["Subscribe"] = d2["Subscribe_Binary"].map({0:"No",1:"Yes"})
        st.plotly_chart(px.box(d2, x="Subscribe", y=col, color="Subscribe",
                        color_discrete_map={"Yes":ACC,"No":"#D6DCE5"}), use_container_width=True)
        rate = d.groupby("Q6_IncomeBand")["Subscribe_Binary"].mean().reset_index()
        st.plotly_chart(px.bar(rate, x="Q6_IncomeBand", y="Subscribe_Binary",
                        title="Subscribe rate by income band", color_discrete_sequence=[PRI]),
                        use_container_width=True)
    with t3:
        corr_cols = ["Q1_Age","IncomeMidpoint_INR","Q7_SavingsRate_Code","Q13_Anxiety",
                     "Q15_LowConfidence","Q17_WantsGuidance","Q18_SelfLiteracy","Q22_WTP_INR",
                     "Q24_TrialLikelihood","PainPoint_Index","FinDiscipline_Index",
                     "AdoptionReadiness","Subscribe_Binary"]
        st.plotly_chart(px.imshow(d[corr_cols].corr().round(2), text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
    with t4:
        cvar = st.selectbox("Categorical", ["Q6_IncomeBand","Q5_Employment","Q23_PricingModel",
                "Q20_GuidanceStyle","Q8_BudgetMethod"])
        ct = pd.crosstab(d[cvar], d["Q25_Subscribe"], normalize="index").round(3)*100
        st.dataframe(ct, use_container_width=True)
        st.plotly_chart(px.bar(ct.reset_index().melt(id_vars=cvar), x=cvar, y="value",
                        color="Q25_Subscribe", barmode="group", labels={"value":"%"}),
                        use_container_width=True)
    st.markdown("#### Static report figures")
    show_fig("eda_histograms.png"); show_fig("eda_corr_heatmap.png")

# ==========================================================================
elif page == "Diagnostic Analysis":
    st.title("Diagnostic Analysis — Drivers of Adoption")
    st.markdown("Statistical tests quantify which factors genuinely influence subscription intent.")
    st.subheader("Numeric drivers — t-tests & effect size (Cohen's d)")
    tt = load_table("diagnostic_ttests.csv")
    if not tt.empty:
        st.dataframe(tt, use_container_width=True, height=320)
        st.plotly_chart(px.bar(tt.sort_values("cohens_d"), x="cohens_d", y="variable",
                        orientation="h", color="cohens_d", color_continuous_scale="RdBu_r",
                        title="Effect size of each driver (subscribers vs non-subscribers)"),
                        use_container_width=True)
    st.subheader("Categorical drivers — Chi-square & Cramér's V")
    ch = load_table("diagnostic_chi_square.csv")
    st.dataframe(ch, use_container_width=True)
    st.subheader("ANOVA across Yes / Maybe / No groups")
    st.dataframe(load_table("diagnostic_anova.csv"), use_container_width=True)
    st.success("Takeaway: trial likelihood, adoption readiness and willingness to pay dominate; "
               "demographics matter far less. Pricing preference and income are the strongest categorical signals.")

# ==========================================================================
elif page == "Customer Segmentation":
    st.title("Customer Segmentation (K-Means)")
    seg = metrics.get("segmentation", {})
    c1,c2 = st.columns(2)
    c1.metric("Optimal clusters (k)", seg.get("best_k","—"))
    c2.metric("Silhouette score", seg.get("silhouette","—"))
    show_fig("seg_elbow_silhouette.png", "Elbow & Silhouette — choosing k")
    c1,c2 = st.columns([1.2,1])
    with c1: show_fig("seg_pca_scatter.png", "Segments in PCA space")
    with c2: show_fig("seg_persona_heatmap.png", "Persona profiles")
    st.subheader("Segment profiles")
    prof = load_table("segment_profiles.csv")
    if not prof.empty:
        st.dataframe(prof, use_container_width=True)
        if "Persona" in df.columns:
            r = df.groupby("Persona")["Subscribe_Binary"].mean().reset_index()
            st.plotly_chart(px.bar(r, x="Persona", y="Subscribe_Binary",
                            title="Subscribe rate by persona", color_discrete_sequence=[PRI]),
                            use_container_width=True)
    st.markdown("""
**Persona strategies**
- **Premium-Ready Professionals** — high income, high intent (~60% subscribe). Lead with micro-investing, AI coach, goal savings; sell the paid plan directly.
- **Financially Stressed Beginners** — large, motivated, cash-constrained. Win with a generous free tier: budgeting, reminders, literacy; monetise later.
- **Smart Savers** — disciplined but price-resistant. Keep on free tier; upsell advanced analytics/credit-score selectively.
""")

# ==========================================================================
elif page == "Machine Learning Models":
    st.title("Machine Learning Models")
    st.markdown(f"Target: **Subscribe_Binary** · {len(features)} predictors · "
                "5 classifiers · stratified 80/20 split · 3-fold CV grid search · class weighting.")
    mc = load_table("model_comparison.csv")
    if not mc.empty:
        st.dataframe(mc, use_container_width=True)
    st.subheader("Confusion matrices"); show_fig("ml_confusion_matrices.png")
    c1,c2 = st.columns(2)
    with c1: show_fig("ml_roc_curves.png", "ROC curves")
    with c2: show_fig("ml_pr_curves.png", "Precision-Recall curves")
    st.subheader("Feature importance"); show_fig("ml_feature_importance.png")

# ==========================================================================
elif page == "Model Comparison":
    st.title("Model Comparison & Selection")
    mc = load_table("model_comparison.csv")
    if not mc.empty:
        metric = st.selectbox("Rank by", ["f1","roc_auc","accuracy","precision","recall","cv_f1"])
        mcs = mc.sort_values(metric, ascending=False)
        st.plotly_chart(px.bar(mcs, x="model", y=metric, color=metric,
                        color_continuous_scale="Blues", title=f"Models ranked by {metric}"),
                        use_container_width=True)
        long = mc.melt(id_vars="model", value_vars=["accuracy","precision","recall","f1","roc_auc"])
        st.plotly_chart(px.line_polar(long, r="value", theta="variable", color="model",
                        line_close=True, title="Metric profile per model"), use_container_width=True)
        show_fig("ml_train_vs_test.png", "Train vs Test accuracy (overfitting check)")
        st.dataframe(mc, use_container_width=True)
        best = mc.sort_values("f1", ascending=False).iloc[0]["model"]
        st.success(f"**Recommended for deployment: {best}** — best balance of F1, precision and ROC-AUC. "
                   "Logistic Regression is the most interpretable companion (highest AUC).")

# ==========================================================================
elif page == "Subscription Prediction":
    st.title("Subscription Prediction")
    st.markdown("Adjust a respondent's profile to estimate their probability of subscribing.")
    model = load_model("best_model.pkl")
    if model is None or not features:
        st.warning("Model artifacts not found. Run run_analysis.py first.")
    else:
        base = df[features].median(numeric_only=True)
        c1,c2,c3 = st.columns(3)
        trial = c1.slider("Free-trial likelihood (Q24)", 1, 5, 3)
        wtp   = c2.slider("Willingness to pay (₹/mo)", 0, 300, 100, step=10)
        income= c3.select_slider("Income (₹/mo)", [15000,27500,42500,65000,100000], 42500)
        c4,c5,c6 = st.columns(3)
        anxiety = c4.slider("Financial anxiety (Q13)", 1, 5, 3)
        literacy= c5.slider("Financial literacy (Q18)", 1, 5, 3)
        guidance= c6.slider("Wants guidance (Q17)", 1, 5, 4)
        c7,c8 = st.columns(2)
        savings = c7.slider("Savings rate rank (0=none..4=>30%)", 0, 4, 2)
        readiness = (trial + 2*(1 if wtp>0 else 0) + guidance)/3

        x = base.copy()
        for k,v in {"Q24_TrialLikelihood":trial,"Q22_WTP_INR":wtp,"IncomeMidpoint_INR":income,
                    "Q13_Anxiety":anxiety,"Q18_SelfLiteracy":literacy,"Q17_WantsGuidance":guidance,
                    "Q7_SavingsRate_Code":savings,"WTP_Positive":1 if wtp>0 else 0,
                    "AdoptionReadiness":readiness}.items():
            if k in x.index: x[k] = v
        X = pd.DataFrame([x])[features]
        proba = float(model.predict_proba(X)[0,1])
        st.markdown("---")
        cc1,cc2 = st.columns([1,1.3])
        cc1.metric("Probability of subscribing", f"{proba*100:.1f}%")
        cc1.write("**Likely to subscribe ✅**" if proba>=0.5 else "**Unlikely to subscribe ❌**")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=proba*100,
                number={"suffix":"%"}, gauge={"axis":{"range":[0,100]},
                "bar":{"color":ACC}, "steps":[{"range":[0,50],"color":"#F2DCDB"},
                {"range":[50,100],"color":"#DCE6F1"}]}))
        gauge.update_layout(height=280, margin=dict(t=10,b=10))
        cc2.plotly_chart(gauge, use_container_width=True)
        st.caption(f"Model: {metrics.get('modeling',{}).get('best_model','best')} · "
                   "prediction uses median values for all other features.")

# ==========================================================================
elif page == "Business Insights":
    st.title("Business Insights & Product Roadmap")
    c1,c2,c3 = st.columns(3)
    c1.metric("Subscribe-ready today", f"{df['Subscribe_Binary'].mean()*100:.1f}%")
    c2.metric("Premium segment conversion", "59.7%")
    c3.metric("Median WTP (payers)", "₹110")
    st.markdown("""
### Who to target first
The ~285 subscription-ready respondents, concentrated in **Premium-Ready Professionals** (≈60% conversion).
Use the model's probability score to rank the rest of the funnel.

### Feature priorities
Micro-investing, goal-based savings, and automated budgeting (most-wanted and payer-aligned).
Literacy content and bill reminders are the hooks for the stressed-beginner majority.

### Pricing
Willingness to pay clusters around **₹100–150/month** among payers atop a large free base.
Recommended: **freemium with a clear monthly/annual upgrade**, anchored near **₹149/mo** —
pricing preference is the single strongest categorical predictor of intent.

### Marketing messages
- Professionals: *"Grow your money on autopilot."*
- Beginners: *"Take control of money stress — free to start."*
- Savers: *"Smarter tools for people who already save."*

### Growth levers
1. **Maximise free-trial starts** — trial likelihood is the #1 predictor.
2. **In-app literacy** to move Beginners up the readiness curve.
3. **Premium investing features** to retain Professionals.
""")
    st.subheader("Suggested roadmap")
    st.table(pd.DataFrame({
        "Phase":["Phase 1 (0–3 mo)","Phase 2 (3–6 mo)","Phase 3 (6–12 mo)"],
        "Focus":["Free tier + frictionless trial; readiness scoring",
                 "Premium tier (micro-investing, AI coach) at ₹149/mo; target Professionals",
                 "Literacy nurture for Beginners; selective upsells; lifecycle propensity model"]}))

# ==========================================================================
elif page == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
The study validates a **viable but focused** opportunity. About **28.5%** of young earners are
subscription-ready today, concentrated in a high-income, high-intent **Premium-Ready Professionals**
segment that converts at ~60%. Adoption is driven by **behavioural readiness and willingness to pay**,
not demographics — so growth depends on frictionless trials, the right premium features, and freemium
pricing near ₹149/month.
A **Random Forest** model predicts subscription intent at **84.5% accuracy (ROC-AUC 0.85)** and is
recommended to score and prioritise leads, with Logistic Regression as an interpretable companion.

**Recommendation:** launch freemium, lead with micro-investing and budgeting, target Premium-Ready
Professionals first, and nurture the large beginner base toward future conversion.

---
*Built with Streamlit · Synthetic dataset for academic analytics · Reproduce via `run_analysis.py`.*
""")
    if metrics.get("modeling"):
        st.dataframe(pd.DataFrame(metrics["modeling"]["comparison"]), use_container_width=True)
