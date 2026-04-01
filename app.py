# =============================================================================
#  Heart Attack Risk Prediction — Streamlit Web Application
#  app.py
#  Run: streamlit run app.py
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSlider > label,
    section[data-testid="stSidebar"] .stSelectbox > label,
    section[data-testid="stSidebar"] .stRadio > label { color: #a8d8ea !important; font-weight: 600; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #c0392b, #8e44ad);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(192,57,43,0.3);
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0 0; font-size: 1.0rem; opacity: 0.88; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
        border-left: 5px solid #c0392b;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { margin: 0.2rem 0 0 0; font-size: 1.9rem; font-weight: 700; color: #2c3e50; }

    /* Risk badges */
    .risk-high     { background:#fee2e2; color:#991b1b; padding:1rem 2rem; border-radius:12px; font-size:1.6rem; font-weight:700; text-align:center; border:2px solid #f87171; }
    .risk-moderate { background:#fef3c7; color:#92400e; padding:1rem 2rem; border-radius:12px; font-size:1.6rem; font-weight:700; text-align:center; border:2px solid #fbbf24; }
    .risk-low      { background:#d1fae5; color:#065f46; padding:1rem 2rem; border-radius:12px; font-size:1.6rem; font-weight:700; text-align:center; border:2px solid #34d399; }

    /* Section headers */
    .section-title { font-size:1.3rem; font-weight:700; color:#1a1a2e; margin:1.5rem 0 0.8rem 0; padding-bottom:0.4rem; border-bottom:3px solid #c0392b; }

    /* Model pills */
    .model-badge { display:inline-block; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.8rem; font-weight:600; margin:0.2rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #c0392b, #8e44ad) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(192,57,43,0.4) !important; }

    /* Warning */
    .disclaimer { background:#fef9c3; border:1px solid #fde68a; border-radius:8px; padding:0.8rem 1rem; font-size:0.82rem; color:#78350f; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
MODEL_DIR = "models"
REPORT_DIR = "reports"

@st.cache_resource
def load_models():
    try:
        scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        models = {
            "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
            "SVM":                 joblib.load(os.path.join(MODEL_DIR, "svm.pkl")),
            "Decision Tree":       joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
            "Random Forest":       joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
        }
        return scaler, feature_names, models, None
    except Exception as e:
        return None, None, None, str(e)

scaler, feature_names, models, load_err = load_models()

NEEDS_SCALING = {
    "Logistic Regression": True, "SVM": True,
    "Decision Tree": False, "Random Forest": False,
}
MODEL_COLORS = {
    "Logistic Regression": "#4C72B0",
    "SVM":                 "#DD8452",
    "Decision Tree":       "#55A868",
    "Random Forest":       "#C44E52",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Attack Risk Predictor</h1>
    <p>AI-powered clinical risk assessment using Logistic Regression, SVM, Decision Tree & Random Forest</p>
</div>
""", unsafe_allow_html=True)

if load_err:
    st.error(f"⚠️ Could not load models: {load_err}  \n"
             f"**Please run `python train_models.py` first.**")
    st.stop()

# ── Sidebar: Patient Input Form ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Information")
    st.markdown("---")

    st.markdown("### 👤 Demographics")
    age    = st.slider("Age (years)", 18, 90, 52)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    bmi    = st.slider("BMI", 15.0, 50.0, 27.5, step=0.1)

    st.markdown("### 🩸 Vitals")
    systolic_bp  = st.slider("Systolic BP (mmHg)", 80, 200, 130)
    diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 120, 85)
    heart_rate   = st.slider("Heart Rate (bpm)", 45, 130, 75)

    st.markdown("### 🧪 Lab Values")
    total_cholesterol = st.slider("Total Cholesterol (mg/dL)", 100, 350, 200)
    ldl  = st.slider("LDL Cholesterol (mg/dL)", 50, 250, 120)
    hdl  = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
    trig = st.slider("Triglycerides (mg/dL)", 50, 600, 130)
    fasting_glucose = st.slider("Fasting Glucose (mg/dL)", 60, 300, 95)
    hba1c    = st.slider("HbA1c (%)", 4.0, 14.0, 5.5, step=0.1)
    crp      = st.slider("CRP (mg/L)", 0.1, 20.0, 1.0, step=0.1)

    st.markdown("### 🚬 Lifestyle")
    smoking  = st.radio("Smoking", ["No", "Yes"], horizontal=True)
    alcohol  = st.selectbox("Alcohol", ["None (0)", "Moderate (1)", "Heavy (2)"])
    physical = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
    diet     = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    stress   = st.slider("Stress Level (1–10)", 1, 10, 5)

    st.markdown("### 🏥 Medical History")
    diabetes    = st.checkbox("Diabetes")
    hypertension= st.checkbox("Hypertension")
    family_hist = st.checkbox("Family History of Heart Disease")
    prev_heart  = st.checkbox("Previous Heart Disease")
    stroke_hist = st.checkbox("Stroke History")
    kidney_dis  = st.checkbox("Kidney Disease")

    st.markdown("### 📊 Clinical Tests")
    ecg_abn = st.checkbox("Abnormal ECG")
    lvh     = st.checkbox("Left Ventricular Hypertrophy")

    st.markdown("### 🤕 Symptoms")
    chest_pain = st.checkbox("Chest Pain")
    sob        = st.checkbox("Shortness of Breath")
    fatigue    = st.checkbox("Fatigue")
    dizziness  = st.checkbox("Dizziness")

    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    selected_model = st.selectbox("Choose Model", list(models.keys()))
    predict_btn    = st.button("🔍 Predict Risk")

# ── Encode input ──────────────────────────────────────────────────────────────
gender_enc   = 1 if gender == "Male" else 0
pa_enc       = {"Low": 1, "Moderate": 2, "High": 0}[physical]
diet_enc     = {"Poor": 2, "Average": 0, "Good": 1}[diet]
alcohol_enc = 1 if alcohol == "Yes" else 0
# alcohol_map = {"Never": 0, "Occasionally": 1, "Regularly": 2}
# alcohol_enc = alcohol_map[alcohol]
smoking_enc  = 1 if smoking == "Yes" else 0

input_dict = {
    "age": age, "gender": gender_enc, "bmi": bmi,
    "smoking": smoking_enc, "alcohol_consumption": alcohol_enc,
    "physical_activity": pa_enc, "diet_quality": diet_enc,
    "stress_level": stress,
    "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp, "heart_rate": heart_rate,
    "total_cholesterol": total_cholesterol, "ldl_cholesterol": ldl,
    "hdl_cholesterol": hdl, "triglycerides": trig,
    "fasting_glucose": fasting_glucose, "hba1c": hba1c, "crp_mg_l": crp,
    "diabetes": int(diabetes), "hypertension": int(hypertension),
    "family_history_heart": int(family_hist), "prev_heart_disease": int(prev_heart),
    "stroke_history": int(stroke_hist), "kidney_disease": int(kidney_dis),
    "ecg_abnormal": int(ecg_abn), "left_ventricular_hypertrophy": int(lvh),
    "chest_pain": int(chest_pain), "shortness_of_breath": int(sob),
    "fatigue": int(fatigue), "dizziness": int(dizziness),
}
input_df = pd.DataFrame([input_dict])[feature_names]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction", "📊 Model Comparison", "📈 Feature Analysis", "ℹ️ About"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        # ── Run all models ────────────────────────────────────────
        all_preds = {}
        for mname, mdl in models.items():
            Xs = scaler.transform(input_df) if NEEDS_SCALING[mname] else input_df
            pred = mdl.predict(Xs)[0]
            prob = mdl.predict_proba(Xs)[0][1]
            all_preds[mname] = {"pred": pred, "prob": prob}

        selected_prob = all_preds[selected_model]["prob"]
        selected_pred = all_preds[selected_model]["pred"]

        # ── Risk level ────────────────────────────────────────────
        if selected_prob >= 0.65:
            risk_level = "HIGH RISK"
            risk_class = "risk-high"
            risk_icon  = "🔴"
        elif selected_prob >= 0.35:
            risk_level = "MODERATE RISK"
            risk_class = "risk-moderate"
            risk_icon  = "🟡"
        else:
            risk_level = "LOW RISK"
            risk_class = "risk-low"
            risk_icon  = "🟢"

        # ── Main result ───────────────────────────────────────────
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.markdown(f'<div class="{risk_class}">{risk_icon} {risk_level}<br>'
                        f'<span style="font-size:1rem;font-weight:400">Risk Probability: {selected_prob*100:.1f}%</span></div>',
                        unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;margin-top:0.5rem;color:#666'>Model used: <b>{selected_model}</b></div>",
                        unsafe_allow_html=True)

        # Gauge chart
        with col_res2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=selected_prob * 100,
                number={"suffix": "%", "font": {"size": 32}},
                delta={"reference": 50, "increasing": {"color": "#c0392b"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkgray"},
                    "bar": {"color": "#c0392b" if selected_pred == 1 else "#2ecc71", "thickness": 0.3},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 35],  "color": "#d1fae5"},
                        {"range": [35, 65], "color": "#fef3c7"},
                        {"range": [65, 100],"color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "black","width": 4}, "thickness": 0.8, "value": 50},
                },
                title={"text": "Heart Attack Risk Score", "font": {"size": 14}},
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── All models comparison ──────────────────────────────────
        st.markdown('<div class="section-title">🤖 All Models Prediction</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (mname, res) in enumerate(all_preds.items()):
            with cols[i]:
                pct   = res["prob"] * 100
                color = MODEL_COLORS[mname]
                label = "⚠️ AT RISK" if res["pred"] == 1 else "✅ NO RISK"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color:{color}">
                    <h3>{mname}</h3>
                    <h2 style="color:{color}">{pct:.1f}%</h2>
                    <p style="margin:0;font-size:0.85rem;font-weight:600">{label}</p>
                </div>""", unsafe_allow_html=True)

        # ── Probability bar ────────────────────────────────────────
        prob_names = list(all_preds.keys())
        prob_vals  = [all_preds[n]["prob"] * 100 for n in prob_names]
        bar_colors = [MODEL_COLORS[n] for n in prob_names]

        fig_bar = go.Figure(go.Bar(
            x=prob_names, y=prob_vals, marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in prob_vals], textposition="outside",
        ))
        fig_bar.add_hline(y=50, line_dash="dash", line_color="black",
                          annotation_text="Decision Threshold (50%)")
        fig_bar.update_layout(
            title="Risk Probability by Model", yaxis_title="Risk Probability (%)",
            yaxis_range=[0, 110], height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Clinical recommendations ──────────────────────────────
        st.markdown('<div class="section-title">💊 Clinical Recommendations</div>', unsafe_allow_html=True)
        recommendations = []

        if systolic_bp >= 140 or diastolic_bp >= 90:
            recommendations.append("🔴 **Hypertension detected** — consult a physician for BP management")
        if total_cholesterol > 240:
            recommendations.append("🔴 **High total cholesterol** — dietary changes and possible medication needed")
        if ldl > 160:
            recommendations.append("🔴 **Elevated LDL** — consider statin therapy evaluation")
        if fasting_glucose > 126 or hba1c > 6.5:
            recommendations.append("🔴 **Diabetes indicators** — endocrinology referral recommended")
        if bmi >= 30:
            recommendations.append("🟡 **Obesity (BMI ≥ 30)** — structured weight management program")
        if smoking == "Yes":
            recommendations.append("🟡 **Smoking cessation** — significantly reduces cardiovascular risk")
        if physical == "Low":
            recommendations.append("🟡 **Low physical activity** — 150 min/week moderate exercise recommended")
        if stress > 7:
            recommendations.append("🟡 **High stress level** — stress management/counselling advised")
        if hdl < 40:
            recommendations.append("🟡 **Low HDL** — increase physical activity, reduce trans fats")
        if family_hist:
            recommendations.append("ℹ️ **Family history present** — more frequent cardiac screening advised")
        if not recommendations:
            recommendations.append("✅ **No critical risk factors identified** — continue healthy lifestyle habits")

        for rec in recommendations:
            st.markdown(f"- {rec}")

        st.markdown('<div class="disclaimer">⚠️ <b>Medical Disclaimer:</b> This tool is for educational/research purposes only. It does NOT replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.</div>',
                    unsafe_allow_html=True)

    else:
        # Default landing state
        st.info("👈 Fill in patient details in the **sidebar** and click **Predict Risk** to generate a heart attack risk assessment.")
        st.markdown("""
        #### How to use this app:
        1. **Enter patient data** in the left sidebar (demographics, vitals, lab values, lifestyle, symptoms)
        2. **Choose a model** from the dropdown
        3. **Click Predict Risk** to generate risk assessment
        4. View **all 4 model predictions** simultaneously
        5. Check **clinical recommendations** generated from input values

        #### Models available:
        | Model | Type | Best for |
        |---|---|---|
        | Logistic Regression | Linear classifier | Interpretable baseline |
        | SVM | Kernel-based | High-dimensional data |
        | Decision Tree | Tree-based | Explainable decisions |
        | Random Forest | Ensemble | High accuracy |
        """)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

    # Load saved comparison CSV
    csv_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    if os.path.exists(csv_path):
        comp_df = pd.read_csv(csv_path, index_col=0)
        display_cols = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC","CV F1 Mean","CV F1 Std"]
        display_df   = comp_df[[c for c in display_cols if c in comp_df.columns]].round(4)

        # Highlight best per column
        styled = display_df.style.highlight_max(axis=0, color="#d1fae5").format("{:.4f}")
        st.dataframe(styled, use_container_width=True)

        # Radar chart
        metrics_radar = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
        available = [m for m in metrics_radar if m in comp_df.columns]
        model_names_r = comp_df.index.tolist()

        fig_radar = go.Figure()
        for mname in model_names_r:
            vals = comp_df.loc[mname, available].tolist()
            vals += [vals[0]]  # close loop
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=available + [available[0]],
                fill="toself", name=mname,
                line_color=MODEL_COLORS.get(mname, "gray"), opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, title="Model Performance Radar Chart",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Grouped bar for all metrics
        fig_gb = go.Figure()
        for metric in available:
            fig_gb.add_trace(go.Bar(
                name=metric,
                x=model_names_r,
                y=comp_df[metric].tolist(),
                text=[f"{v:.3f}" for v in comp_df[metric]],
                textposition="outside",
            ))
        fig_gb.update_layout(
            barmode="group", title="All Metrics — All Models",
            yaxis_title="Score", yaxis_range=[0, 1.12], height=400,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gb, use_container_width=True)

    else:
        st.warning("Run `python train_models.py` to generate comparison data.")

    # Show saved plots
    for img_file, title in [
        ("roc_curves.png", "ROC Curves"),
        ("pr_curves.png", "Precision-Recall Curves"),
        ("cv_boxplot.png", "Cross-Validation Scores"),
        ("confusion_matrices.png", "Confusion Matrices"),
    ]:
        path = os.path.join(REPORT_DIR, img_file)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_column_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📈 Feature Importance & Patient Profile</div>', unsafe_allow_html=True)

    # Show saved feature importance plot
    fi_path = os.path.join(REPORT_DIR, "feature_importance.png")
    if os.path.exists(fi_path):
        st.image(fi_path, use_column_width=True)

    # Live patient feature radar
    if predict_btn:
        st.markdown("#### Your Patient's Risk Factor Profile")

        # Normalize a few key features for radar display
        radar_features = {
            "Age": min(age / 90, 1.0),
            "BMI": min(bmi / 45, 1.0),
            "Systolic BP": min(systolic_bp / 200, 1.0),
            "LDL": min(ldl / 250, 1.0),
            "Glucose": min(fasting_glucose / 300, 1.0),
            "HbA1c": min(hba1c / 14, 1.0),
            "CRP": min(crp / 20, 1.0),
            "Stress": stress / 10,
            "Triglycerides": min(trig / 600, 1.0),
        }
        labels = list(radar_features.keys())
        values = list(radar_features.values()) + [list(radar_features.values())[0]]
        labels_c = labels + [labels[0]]

        fig_pt = go.Figure()
        fig_pt.add_trace(go.Scatterpolar(
            r=values, theta=labels_c, fill="toself",
            name="Patient", line_color="#c0392b", fillcolor="rgba(192,57,43,0.2)"
        ))
        fig_pt.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Patient Risk Factor Radar (Normalised)", height=420,
        )
        st.plotly_chart(fig_pt, use_container_width=True)

        # Binary risk factors table
        st.markdown("#### Binary Risk Factors Summary")
        binary_data = {
            "Risk Factor": ["Diabetes", "Hypertension", "Family History",
                            "Prev. Heart Disease", "Stroke History", "Kidney Disease",
                            "Abnormal ECG", "LV Hypertrophy",
                            "Chest Pain", "Shortness of Breath", "Fatigue", "Dizziness",
                            "Smoking"],
            "Present": [diabetes, hypertension, family_hist, prev_heart,
                        stroke_hist, kidney_dis, ecg_abn, lvh,
                        chest_pain, sob, fatigue, dizziness, smoking == "Yes"],
        }
        b_df = pd.DataFrame(binary_data)
        b_df["Status"] = b_df["Present"].map({True: "✅ Yes", False: "❌ No"})
        risk_count = b_df["Present"].sum()
        st.dataframe(b_df[["Risk Factor","Status"]], use_container_width=True)
        st.metric("Total Risk Factors Present", f"{risk_count} / {len(b_df)}")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">ℹ️ About This Application</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Overview
    This application uses machine learning models trained on a **20,000-patient synthetic dataset** to predict
    the risk of a heart attack based on clinical, demographic, and lifestyle factors.

    ### Dataset Features (31 input features)
    | Category | Features |
    |---|---|
    | **Demographics** | Age, Gender, BMI |
    | **Vitals** | Systolic BP, Diastolic BP, Heart Rate |
    | **Lab Values** | Total Cholesterol, LDL, HDL, Triglycerides, Fasting Glucose, HbA1c, CRP |
    | **Lifestyle** | Smoking, Alcohol, Physical Activity, Diet Quality, Stress Level |
    | **Medical History** | Diabetes, Hypertension, Family History, Prev. Heart Disease, Stroke, Kidney Disease |
    | **Clinical Tests** | ECG Abnormality, Left Ventricular Hypertrophy |
    | **Symptoms** | Chest Pain, Shortness of Breath, Fatigue, Dizziness |

    ### Models
    | Model | Description |
    |---|---|
    | **Logistic Regression** | Linear model; highly interpretable, fast, good baseline |
    | **SVM (RBF kernel)** | Non-linear decision boundary; robust on normalised data |
    | **Decision Tree** | Rule-based; fully explainable; prone to overfitting if not tuned |
    | **Random Forest** | Ensemble of 200 trees; typically best accuracy & robustness |

    ### Workflow
    ```
    1. python train_models.py   →  trains all 4 models, saves to models/
    2. python model_test.py     →  evaluates on held-out test set, saves plots to reports/
    3. streamlit run app.py     →  launches this web application
    ```

    ### Risk Thresholds
    | Risk Level | Probability |
    |---|---|
    | 🟢 Low Risk | < 35% |
    | 🟡 Moderate Risk | 35% – 65% |
    | 🔴 High Risk | > 65% |

    ---
    > ⚠️ **Disclaimer**: This tool is for **research and educational purposes only**.
    > It should NOT be used as a substitute for professional medical diagnosis or treatment.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Dataset Size", "20,000 patients")
        st.metric("Features", "31 input variables")
        st.metric("Train / Test Split", "80% / 20%")
    with col_b:
        st.metric("Cross-Validation", "5-fold Stratified K-Fold")
        st.metric("Models Compared", "4")
        st.metric("Evaluation Metrics", "Accuracy, F1, AUC, Precision, Recall")
