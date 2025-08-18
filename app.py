import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Employee Attrition – Dashboard & Predictor", layout="wide")

st.title("📊 Employee Attrition & Performance Predictor")

# ---- Load artifacts ----
data_path = Path("data/cleaned_data.csv")
attr_model_path = Path("model/model.pkl")       # Attrition Model
perf_model_path = Path("model/perf_model.pkl")     # Performance Model

@st.cache_data
def load_data(path):
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_model(path):
    if path.exists():
        return joblib.load(path)
    return None

df = load_data(data_path)
attr_model = load_model(attr_model_path)
perf_model = load_model(perf_model_path)

if df is None or attr_model is None or perf_model is None:
    st.error("⚠️ Missing files: Please place `employee_attrition_cleaned.csv`, `rf_model.pkl`, and `perf_model.pkl` in the same folder as this app.")
    st.stop()

# Columns used by the models (make sure consistent with training)
feature_cols = [c for c in df.columns if c not in ["Attrition", "PerformanceRating"]]

# Defaults for inputs
defaults = df[feature_cols].median(numeric_only=True).to_dict()
for c in feature_cols:
    if c not in defaults:
        defaults[c] = df[c].mode().iloc[0]

# ---- Sidebar Filters ----
with st.sidebar:
    st.header("🔎 Overview Filters")
    filt = pd.Series([True] * len(df))

    for col in ["Department", "JobRole", "OverTime"]:
        if col in df.columns:
            vals = sorted(df[col].unique().tolist())
            sel = st.multiselect(col, vals, default=vals)
            filt &= df[col].isin(sel)

tab1, tab2, tab3 = st.tabs(["📈 Overview", "🤖 Attrition Predictor", "⭐ Performance Predictor"])

# ---- Overview Tab ----
with tab1:
    st.subheader("KPIs")
    sub = df[filt]

    total = len(sub)
    attr_count = int(sub["Attrition"].sum()) if "Attrition" in sub.columns else 0
    attr_rate = (attr_count / total * 100) if total else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Employees (filtered)", f"{total:,}")
    c2.metric("Attrition Count", f"{attr_count:,}")
    c3.metric("Attrition Rate", f"{attr_rate:.1f}%")

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if "MonthlyIncome" in sub.columns:
            st.bar_chart(sub.groupby("Attrition")["MonthlyIncome"].median())
    with colB:
        if "Age" in sub.columns:
            st.line_chart(sub.groupby("Age")["Attrition"].mean())

# ---- Attrition Predictor ----
with tab2:
    st.subheader("Single Employee – Attrition Risk")

    ui_cols = [c for c in ["Age","DistanceFromHome","MonthlyIncome","YearsAtCompany","JobSatisfaction",
                           "OverTime","WorkLifeBalance","JobLevel","TotalWorkingYears","YearsSinceLastPromotion"]
               if c in feature_cols]

    user_vals = {}
    left, right = st.columns(2)
    for idx, c in enumerate(ui_cols):
        target_col = left if idx % 2 == 0 else right
        with target_col:
            series = df[c]
            default = defaults.get(c, series.median())

            if series.dtype.kind in "biu":  # integer
                mn, mx = int(series.min()), int(series.max())
                if c.lower() == "overtime" and set(series.unique()) <= {0,1}:
                    yn = st.selectbox("OverTime", ["No","Yes"], index=1 if default==1 else 0)
                    val = 1 if yn == "Yes" else 0
                else:
                    val = st.slider(c, mn, mx, int(default))
                user_vals[c] = int(val)
            else:  # float
                mn, mx = float(series.min()), float(series.max())
                val = st.number_input(c, mn, mx, float(default))
                user_vals[c] = float(val)

    # Build feature vector
    x = {col: defaults.get(col, 0) for col in feature_cols}
    x.update(user_vals)

    if st.button("Predict Attrition Risk"):
        X = pd.DataFrame([x])[feature_cols]
        proba = attr_model.predict_proba(X)[0][1]
        pred = int(proba >= 0.5)

        if pred == 1:
            st.error(f"⚠️ High Attrition Risk — Probability: {proba*100:.2f}%")
        else:
            st.success(f"✅ Low Attrition Risk — Probability: {proba*100:.2f}%")

# ---- Performance Predictor ----
with tab3:
    st.subheader("Single Employee – Performance Rating Prediction")

    ui_cols_perf = [c for c in ["Education","JobInvolvement","JobLevel","MonthlyIncome",
                                "YearsAtCompany","YearsInCurrentRole","TotalWorkingYears"]
                    if c in feature_cols]

    user_vals_perf = {}
    left, right = st.columns(2)
    for idx, c in enumerate(ui_cols_perf):
        target_col = left if idx % 2 == 0 else right
        with target_col:
            series = df[c]
            default = defaults.get(c, series.median())

            if series.dtype.kind in "biu":  # integer
                mn, mx = int(series.min()), int(series.max())
                val = st.slider(c, mn, mx, int(default))
                user_vals_perf[c] = int(val)
            else:
                mn, mx = float(series.min()), float(series.max())
                val = st.number_input(c, mn, mx, float(default))
                user_vals_perf[c] = float(val)

    x_perf = {col: defaults.get(col, 0) for col in feature_cols}
    x_perf.update(user_vals_perf)

    if st.button("Predict Performance Rating"):
        Xp = pd.DataFrame([x_perf])[feature_cols]
        pred_perf = perf_model.predict(Xp)[0]
        st.info(f"⭐ Predicted Performance Rating: {int(pred_perf)} (scale 1–4)")
        st.markdown("**Note:** Higher ratings indicate better performance.")