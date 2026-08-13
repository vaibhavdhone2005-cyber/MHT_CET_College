import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

# Page Config
st.set_page_config(
    page_title="Visionary Predictor Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #E2E8F0;
    }

    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #020617);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .gradient-text {
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find model files in project root
def get_file_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)

@st.cache_resource
def load_models():
    model1, model2 = None, None
    m1_path = get_file_path("model1.pkl")
    m2_path = get_file_path("model2.pkl")

    if os.path.exists(m1_path):
        with open(m1_path, "rb") as f:
            model1 = pickle.load(f)

    if os.path.exists(m2_path):
        with open(m2_path, "rb") as f:
            model2 = pickle.load(f)

    return model1, model2

model1, model2 = load_models()

# Fallback feature list extraction
feature_names = ["Merit Number", "MHTCET Percentile", "Gender", "Category"]
if model1 and hasattr(model1, "feature_names_in_"):
    feature_names = list(model1.feature_names_in_)

# Application Header
st.markdown("<h1 style='text-align: center;' class='gradient-text'>⚡ AI Prediction & Analysis Portal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 40px;'>Multi-Model Analytical Dashboard for Admission & Seat Allocation Forecasting</p>", unsafe_allow_html=True)

# Main UI
col_input, col_display = st.columns([1, 1.6], gap="large")

with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #A855F7;'>🎛️ Input Parameters</h3>", unsafe_allow_html=True)
    
    user_inputs = {}
    for feature in feature_names:
        if "percentile" in feature.lower():
            user_inputs[feature] = st.number_input(f"Enter {feature}", min_value=0.0, max_value=100.0, value=92.5, step=0.01)
        elif any(k in feature.lower() for k in ["merit", "number", "rank"]):
            user_inputs[feature] = st.number_input(f"Enter {feature}", min_value=1, max_value=200000, value=5000, step=1)
        elif "gender" in feature.lower():
            gender_val = st.selectbox(f"Select {feature}", ["Male", "Female", "Other"])
            user_inputs[feature] = 1 if gender_val == "Male" else (0 if gender_val == "Female" else 2)
        elif "category" in feature.lower():
            cat_val = st.selectbox(f"Select {feature}", ["OPEN", "OBC", "SC", "ST", "NT/VJ", "EWS"])
            category_mapping = {"OPEN": 0, "OBC": 1, "SC": 2, "ST": 3, "NT/VJ": 4, "EWS": 5}
            user_inputs[feature] = category_mapping[cat_val]
        else:
            user_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)
            
    predict_btn = st.button("🚀 Run Multi-Model Prediction")
    st.markdown('</div>', unsafe_allow_html=True)

with col_display:
    if predict_btn:
        input_data = np.array([list(user_inputs.values())])
        
        pred1, proba1 = "N/A", 100.0
        pred2, proba2 = "N/A", 100.0
        
        if model1:
            pred1 = model1.predict(input_data)[0]
            if hasattr(model1, "predict_proba"):
                proba1 = float(np.max(model1.predict_proba(input_data)) * 100)
                
        if model2:
            pred2 = model2.predict(input_data)[0]
            if hasattr(model2, "predict_proba"):
                proba2 = float(np.max(model2.predict_proba(input_data)) * 100)
        else:
            pred2 = pred1
            proba2 = proba1

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #EC4899;'>🎯 Prediction Results Summary</h3>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("#### 🔹 Model 1 Outcome")
            st.info(f"**Result:**\n{pred1}")
            st.caption(f"Confidence: {proba1:.2f}%")

        with res_col2:
            st.markdown("#### 🔸 Model 2 Outcome")
            st.success(f"**Result:**\n{pred2}")
            st.caption(f"Confidence: {proba2:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #6366F1;'>📊 Insights</h3>", unsafe_allow_html=True)
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=['Model 1', 'Model 2'],
                y=[proba1, proba2],
                marker_color=['#6366f1', '#ec4899'],
                text=[f"{proba1:.1f}%", f"{proba2:.1f}%"],
                textposition='auto',
            )
        ])
        fig_bar.update_layout(
            title="Model Confidence Score Comparison",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#E2E8F0"),
            yaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👈 Fill in the parameters and click **Run Multi-Model Prediction**.")
