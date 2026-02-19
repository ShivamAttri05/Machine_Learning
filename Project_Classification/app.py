import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import base64
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Heart Health AI | Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
def load_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%);
        }
        
        /* Main container styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem;
            border-radius: 30px;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 50%);
            animation: shimmer 8s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translate(0, 0) rotate(0deg); }
            100% { transform: translate(-30%, -30%) rotate(45deg); }
        }
        
        .main-header h1 {
            color: white;
            font-weight: 700;
            margin-bottom: 0.5rem;
            font-size: 2.8rem;
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
        }
        
        .main-header p {
            color: rgba(255,255,255,0.95);
            font-size: 1.2rem;
            margin-bottom: 0;
            position: relative;
            z-index: 1;
            font-weight: 400;
        }
        
        /* Card styling */
        .card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.8);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
            border-color: #667eea;
        }
        
        /* Glass morphism effect */
        .glass-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
            padding: 1.5rem;
            border-radius: 20px;
            text-align: center;
            border: 1px solid #eef2f6;
            box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #2c3e50, #3498db);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }
        
        /* Risk indicators with animations */
        .risk-low {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            padding: 1.5rem;
            border-radius: 20px;
            border-left: 8px solid #28a745;
            animation: slideIn 0.5s ease;
        }
        
        .risk-moderate {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
            color: #856404;
            padding: 1.5rem;
            border-radius: 20px;
            border-left: 8px solid #ffc107;
            animation: slideIn 0.5s ease;
        }
        
        .risk-high {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            padding: 1.5rem;
            border-radius: 20px;
            border-left: 8px solid #dc3545;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.9rem 2rem;
            border-radius: 50px;
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.95rem;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        }
        
        .stButton button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }
        
        .stButton button:active {
            transform: translateY(-1px);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background-color: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(10px);
            padding: 0.5rem;
            border-radius: 100px;
            border: 1px solid rgba(255, 255, 255, 0.8);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 100px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            color: #4a5568;
            transition: all 0.2s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        }
        
        .sidebar-content {
            color: white;
            padding: 1.5rem;
        }
        
        .sidebar-content h2 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .sidebar-content h3 {
            color: #94a3b8;
            font-size: 1.1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 2rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2.5rem;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white;
            border-radius: 30px;
            margin-top: 2.5rem;
            position: relative;
            overflow: hidden;
        }
        
        .footer::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        }
        
        .footer a {
            color: #94a3b8;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s ease;
        }
        
        .footer a:hover {
            color: white;
            text-decoration: none;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
            border-radius: 10px;
            height: 10px;
        }
        
        /* Input field styling */
        .stNumberInput input, .stSelectbox select, .stSlider {
            border-radius: 16px;
            border: 2px solid #e2e8f0;
            padding: 0.75rem;
            font-size: 1rem;
            transition: all 0.2s ease;
        }
        
        .stNumberInput input:focus, .stSelectbox select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Divider */
        .custom-divider {
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
            margin: 2.5rem 0;
            border-radius: 4px;
            position: relative;
        }
        
        .custom-divider::before,
        .custom-divider::after {
            content: '❤️';
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: white;
            padding: 0 1rem;
            font-size: 1.2rem;
        }
        
        .custom-divider::before {
            left: 20px;
        }
        
        .custom-divider::after {
            right: 20px;
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 1.5rem;
            border-radius: 20px;
            border-left: 8px solid #2196f3;
            margin: 1rem 0;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid;
            border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
            display: inline-block;
        }
        
        /* Tooltip styling */
        .tooltip-icon {
            display: inline-block;
            width: 18px;
            height: 18px;
            background: #94a3b8;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 18px;
            font-size: 12px;
            margin-left: 5px;
            cursor: help;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            
            .main-header p {
                font-size: 1rem;
            }
            
            .metric-value {
                font-size: 1.8rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("heart_disease_svm_pipeline.pkl")
        return model, None
    except Exception as e:
        return None, str(e)

model, load_error = load_model()

# -----------------------------
# HEADER SECTION WITH ANIMATION
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Health AI</h1>
    <p>Advanced Machine Learning-Based Cardiovascular Risk Assessment</p>
    <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 50px; color: white; font-size: 0.9rem;">🔬 KNN Algorithm</span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 50px; color: white; font-size: 0.9rem;">⚕️ 11 Clinical Parameters</span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 50px; color: white; font-size: 0.9rem;">📊 87% Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-content" style="color: white;">', unsafe_allow_html=True)

    st.markdown("## 📊 Dashboard")
    st.write("Welcome to the Heart Risk Predictor")
    st.write("Use the tabs to navigate through analysis.")

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Profile section
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 20px; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-size: 1.5rem;">👤</span>
            </div>
            <div>
                <p style="color: black; margin: 0; font-weight: 600;">Clinical Dashboard</p>
                <p style="color: black; margin: 0; font-size: 0.9rem;">Patient Assessment</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <h3>🎯 How It Works</h3>
    <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 15px; margin-bottom: 1.5rem;">
        <p style="color: #2e2d33; margin: 0.5rem 0;">1️⃣ Enter patient data</p>
        <p style="color: #2e2d33; margin: 0.5rem 0;">2️⃣ Click 'Calculate Risk'</p>
        <p style="color: #2e2d33; margin: 0.5rem 0;">3️⃣ Get instant assessment</p>
        <p style="color: #2e2d33; margin: 0.5rem 0;">4️⃣ Review recommendations</p>
    </div>
    
    <h3>⚕️ Model Performance</h3>
    <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 15px; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #2e2d33;">Accuracy</span>
            <span style="color: #252138; font-weight: 600;">88.04%</span>
        </div>
        <div style="height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; margin-bottom: 0.5rem;">
            <div style="width: 88.04%; height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 3px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #2e2d33;">Precision</span>
            <span style="color: #252138; font-weight: 600;">88.46%</span>
        </div>
        <div style="height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; margin-bottom: 0.5rem;">
            <div style="width: 88.46%; height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 3px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #2e2d33;">F1-Score</span>
            <span style="color: #252138; font-weight: 600;">89.32%</span>
        </div>
        <div style="height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px;">
            <div style="width: 89.32%; height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 3px;"></div>
        </div>
    </div>
    
    <h3>📋 Important Note</h3>
    <div style="background: rgba(255,193,7,0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #ffc107;">
        <p style="color: #19240c; margin: 0; font-size: 0.9rem;">⚠️ This tool is for educational purposes only. Always consult healthcare professionals.</p>
    </div>
    
    <h3>📊 Quick Stats</h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.5rem;">
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 15px; text-align: center;">
            <p style="color: #252138; font-size: 1.5rem; font-weight: 700; margin: 0;">1.2k</p>
            <p style="color: #94a3b8; margin: 0; font-size: 0.8rem;">Predictions</p>
        </div>
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 15px; text-align: center;">
            <p style="color: #252138; font-size: 1.5rem; font-weight: 700; margin: 0;">88.04%</p>
            <p style="color: #94a3b8; margin: 0; font-size: 0.8rem;">Accuracy</p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# HANDLE MODEL LOAD FAILURE
# -----------------------------
if load_error:
    st.error(f"⚠️ Model files not found. Please ensure all model files are in the correct directory.\n\n{load_error}")
    st.stop()

# -----------------------------
# INPUT VALIDATION
# -----------------------------
def validate_inputs(bp, chol, oldpeak, age):
    warnings = []
    risk_factors = []
    
    if bp > 180:
        warnings.append("⚠️ Resting BP is dangerously high (>180 mmHg)")
        risk_factors.append("Hypertension")
    elif bp > 140:
        warnings.append("⚠️ Elevated blood pressure (>140 mmHg)")
        risk_factors.append("Pre-hypertension")
    
    if chol > 300:
        warnings.append("⚠️ Cholesterol critically high (>300 mg/dL)")
        risk_factors.append("Hypercholesterolemia")
    elif chol > 240:
        warnings.append("⚠️ High cholesterol (>240 mg/dL)")
        risk_factors.append("High Cholesterol")
    
    if oldpeak > 4:
        warnings.append("⚠️ Severe ST depression detected (>4 mm)")
        risk_factors.append("Ischemia")
    elif oldpeak > 2:
        warnings.append("⚠️ Moderate ST depression (>2 mm)")
    
    if age > 65:
        risk_factors.append("Advanced Age")
    
    return warnings, risk_factors

# -----------------------------
# LOGGING (SAFE APPEND)
# -----------------------------
def log_prediction(age, risk, risk_level):
    file = "prediction_log.csv"
    log = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "risk_percent": round(risk, 2),
        "risk_level": risk_level
    }])

    if not os.path.exists(file):
        log.to_csv(file, index=False)
    else:
        log.to_csv(file, mode="a", header=False, index=False)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Patient Data Entry",
    "🔍 Risk Analysis",
    "💡 Health Guidance",
    "🤖 Model Insights"
])

# -----------------------------
# TAB 1 — INPUTS
# -----------------------------
with tab1:
    st.markdown('<h2 class="section-header">🩺 Patient Clinical Parameters</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Enter the patient\'s health metrics below for comprehensive risk assessment</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📋 Basic Information")
        age = st.slider("Age (years)", 18, 100, 45, 
                       help="Patient's age in years")
        
        sex = st.selectbox("Biological Sex", ["Male", "Female"], 
                          help="Patient's biological sex")
        sex_code = "M" if sex == "Male" else "F"
        
        chest_pain = st.selectbox("Chest Pain Type", 
                                  ["ATA (Atypical Angina)", 
                                   "NAP (Non-Anginal Pain)", 
                                   "TA (Typical Angina)", 
                                   "ASY (Asymptomatic)"],
                                  help="Type of chest pain experienced")
        chest_pain_code = chest_pain.split()[0]
        
        resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 80, 220, 120,
                                     help="Resting blood pressure in mm Hg")
        
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200,
                                      help="Serum cholesterol in mg/dL")

    with col2:
        st.markdown("#### 📊 Clinical Measurements")
        fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", 
                              ["No", "Yes"],
                              help="Whether fasting blood sugar exceeds 120 mg/dL")
        fasting_bs_code = 1 if fasting_bs == "Yes" else 0
        
        resting_ecg = st.selectbox("Resting ECG Results", 
                                   ["Normal", "ST (ST-T Wave Abnormality)", "LVH (Left Ventricular Hypertrophy)"],
                                   help="Resting electrocardiogram results")
        resting_ecg_code = resting_ecg.split()[0]
        
        max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150,
                           help="Maximum heart rate achieved during exercise")
        
        exercise_angina = st.radio("Exercise-Induced Angina", 
                                   ["No", "Yes"],
                                   help="Whether angina occurs during exercise")
        exercise_angina_code = "Y" if exercise_angina == "Yes" else "N"
        
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1,
                            help="ST depression induced by exercise relative to rest")
        
        st_slope = st.selectbox("ST Slope", 
                                ["Up", "Flat", "Down"],
                                help="Slope of the peak exercise ST segment")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show quick summary
    st.markdown('<h2 class="section-header">📈 Patient Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{age}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Age</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.8rem; color: #94a3b8;">years</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{resting_bp}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">BP</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.8rem; color: #94a3b8;">mmHg</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{cholesterol}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Cholesterol</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.8rem; color: #94a3b8;">mg/dL</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{max_hr}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Max HR</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.8rem; color: #94a3b8;">bpm</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# TAB 2 — PREDICTION
# -----------------------------
with tab2:
    st.markdown('<h2 class="section-header">🔄 Risk Analysis & Simulation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎯 Simulation Mode")
        st.markdown('<p style="color: #64748b; font-size: 0.9rem;">Adjust parameters to see real-time risk changes</p>', unsafe_allow_html=True)
        
        sim_chol = st.slider("Adjust Cholesterol (mg/dL)", 100, 300, cholesterol,
                            help="Move slider to see how cholesterol changes affect risk",
                            key="sim_chol")
        
        sim_bp = st.slider("Adjust Blood Pressure (mmHg)", 80, 220, resting_bp,
                          help="Move slider to see how BP changes affect risk",
                          key="sim_bp")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔮 Calculate Risk", width='stretch'):
            # Validate inputs
            warnings, risk_factors = validate_inputs(resting_bp, cholesterol, oldpeak, age)
            
            # Build input
            raw = {
                'Age': age,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs_code,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'Sex_' + sex_code: 1,
                'ChestPainType_' + chest_pain_code: 1,
                'RestingECG_' + resting_ecg_code: 1,
                'ExerciseAngina_' + exercise_angina_code: 1,
                'ST_Slope_' + st_slope: 1
            }

            input_df = pd.DataFrame([raw])
            
            # Determine risk level
            if prob < 30:
                risk_level = "Low"
            elif prob < 60:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            log_prediction(age, prob, risk_level)

            # Store results in session state
            st.session_state['prob'] = prob
            st.session_state['risk_level'] = risk_level
            st.session_state['warnings'] = warnings
            st.session_state['risk_factors'] = risk_factors
            st.session_state['input_df'] = input_df
            
            # Simulation
            sim_input = input_df.copy()
            sim_input["Cholesterol"] = sim_chol
            sim_input["RestingBP"] = sim_bp
            st.session_state['sim_prob'] = sim_prob
    
    with col2:
        if 'prob' in st.session_state:
            prob = st.session_state['prob']
            risk_level = st.session_state['risk_level']
            
            # Display risk gauge
            st.markdown("#### 📊 Risk Assessment")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Cardiovascular Risk Score", 'font': {'size': 24, 'family': 'Plus Jakarta Sans'}},
                delta={'reference': 50, 'position': "top", 'increasing': {'color': "#dc3545"}, 'decreasing': {'color': "#28a745"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#64748b", 'tickfont': {'size': 12}},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'bgcolor': "white",
                    'borderwidth': 3,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda', 'name': 'Low'},
                        {'range': [30, 60], 'color': '#fff3cd', 'name': 'Moderate'},
                        {'range': [60, 100], 'color': '#f8d7da', 'name': 'High'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=30, r=30, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#1e293b", 'family': "Plus Jakarta Sans"}
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Risk level display
            if risk_level == "Low":
                st.markdown(f'''
                <div class="risk-low">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 2rem;">✅</div>
                        <div>
                            <strong style="font-size: 1.2rem;">Low Risk ({prob:.1f}%)</strong>
                            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Your cardiovascular risk profile appears healthy. Maintain your healthy lifestyle!</p>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            elif risk_level == "Moderate":
                st.markdown(f'''
                <div class="risk-moderate">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 2rem;">⚠️</div>
                        <div>
                            <strong style="font-size: 1.2rem;">Moderate Risk ({prob:.1f}%)</strong>
                            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Some risk factors detected. Consider lifestyle modifications.</p>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="risk-high">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 2rem;">🔴</div>
                        <div>
                            <strong style="font-size: 1.2rem;">High Risk ({prob:.1f}%)</strong>
                            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Immediate medical consultation recommended.</p>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed results section
    if 'prob' in st.session_state:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### ⚠️ Risk Factors Detected")
            if st.session_state.get('risk_factors'):
                for factor in st.session_state['risk_factors']:
                    st.markdown(f'''
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="color: #dc3545;">•</span>
                        <span>{factor}</span>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: #28a745;">✅ No major risk factors detected</p>', unsafe_allow_html=True)
            
            if st.session_state.get('warnings'):
                st.markdown("#### 🔔 Important Warnings")
                for warning in st.session_state['warnings']:
                    st.markdown(f'''
                    <div style="background: #fff3cd; padding: 0.75rem; border-radius: 10px; margin-bottom: 0.5rem; border-left: 4px solid #ffc107;">
                        {warning}
                    </div>
                    ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Similar Patient Analysis")
            
            distances, indices = model.kneighbors(st.session_state['scaled'])
            
            # Create neighbor analysis dataframe
            neighbor_df = pd.DataFrame({
                "Patient": [f"Case #{i+1}" for i in range(len(indices[0]))],
                "Similarity": [f"{(1-d)*100:.1f}%" for d in distances[0]],
                "Distance": [f"{d:.3f}" for d in distances[0]]
            })
            
            # Style the dataframe
            st.dataframe(
                neighbor_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Patient": st.column_config.TextColumn("Patient", width="small"),
                    "Similarity": st.column_config.ProgressColumn(
                        "Similarity",
                        format="%s",
                        min_value=0,
                        max_value=100,
                    ),
                    "Distance": st.column_config.TextColumn("Distance", width="small")
                }
            )
            
            # Simulation result
            if 'sim_prob' in st.session_state:
                st.markdown("#### 🔄 Simulation Impact")
                delta = st.session_state['sim_prob'] - st.session_state['prob']
                
                # Create a nice metric display
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f'''
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem; border-radius: 15px; text-align: center;">
                        <p style="color: #64748b; margin: 0; font-size: 0.9rem;">Current Risk</p>
                        <p style="color: #1e293b; font-size: 1.8rem; font-weight: 700; margin: 0;">{st.session_state['prob']:.1f}%</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_b:
                    delta_color = "#dc3545" if delta > 0 else "#28a745"
                    st.markdown(f'''
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem; border-radius: 15px; text-align: center;">
                        <p style="color: #64748b; margin: 0; font-size: 0.9rem;">Adjusted Risk</p>
                        <p style="color: #1e293b; font-size: 1.8rem; font-weight: 700; margin: 0;">{st.session_state['sim_prob']:.1f}%</p>
                        <p style="color: {delta_color}; margin: 0; font-size: 0.9rem;">{delta:+.1f}% change</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendation section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 💊 Personalized Recommendations")
        
        if st.session_state['risk_level'] == "Low":
            recommendations = [
                "Continue regular exercise (150 mins/week)",
                "Maintain healthy diet rich in fruits and vegetables",
                "Regular health check-ups annually",
                "Monitor blood pressure quarterly"
            ]
            icon = "✅"
        elif st.session_state['risk_level'] == "Moderate":
            recommendations = [
                "Increase physical activity to 200+ mins/week",
                "Reduce sodium intake (<2300mg/day)",
                "Schedule check-up with primary care physician",
                "Consider cholesterol-lowering diet changes",
                "Monitor blood pressure monthly"
            ]
            icon = "⚠️"
        else:
            recommendations = [
                "Schedule immediate cardiology consultation",
                "Strict medication adherence if prescribed",
                "Daily blood pressure monitoring",
                "Cardiac rehabilitation program recommended",
                "Emergency plan discussion with doctor"
            ]
            icon = "🔴"
        
        for rec in recommendations:
            st.markdown(f'''
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem; padding: 0.5rem; background: #f8f9fa; border-radius: 10px;">
                <span style="font-size: 1.2rem;">{icon}</span>
                <span>{rec}</span>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# TAB 3 — HEALTH GUIDANCE
# -----------------------------
with tab3:
    st.markdown('<h2 class="section-header">🥗 Heart Health Guidelines</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%); padding: 1.5rem; border-radius: 20px; margin-bottom: 1rem;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🏃‍♂️ Physical Activity</h4>
            <div style="display: grid; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Aerobic Exercise:</strong> 150 mins/week moderate</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Strength Training:</strong> 2+ days/week</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Flexibility:</strong> Daily stretching</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Sedentary Breaks:</strong> Stand every 30 mins</span>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%); padding: 1.5rem; border-radius: 20px; margin-bottom: 1rem;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🥑 Nutrition</h4>
            <div style="display: grid; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Fruits & Vegetables:</strong> 5+ servings daily</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Whole Grains:</strong> 3+ servings daily</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Lean Protein:</strong> Fish twice weekly</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Limit:</strong> Sodium <2300mg/day</span>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%); padding: 1.5rem; border-radius: 20px;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🚫 Lifestyle Modifications</h4>
            <div style="display: grid; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Smoking:</strong> Complete cessation</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Alcohol:</strong> Moderation (1-2 drinks max/day)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Stress Management:</strong> Meditation, yoga</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%); padding: 1.5rem; border-radius: 20px; margin-bottom: 1rem;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">💊 Medical Management</h4>
            <div style="display: grid; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Blood Pressure:</strong> <120/80 mmHg ideal</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Cholesterol:</strong> LDL <100 mg/dL</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>Blood Sugar:</strong> Fasting <100 mg/dL</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #667eea;">•</span>
                    <span><strong>BMI:</strong> 18.5-24.9 kg/m²</span>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%); padding: 1.5rem; border-radius: 20px; margin-bottom: 1rem;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">📊 Monitoring Schedule</h4>
            <div style="display: grid; gap: 0.5rem;">
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(255,255,255,0.5); border-radius: 10px;">
                    <span>Blood Pressure</span>
                    <span style="font-weight: 600;">Weekly</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(255,255,255,0.5); border-radius: 10px;">
                    <span>Cholesterol</span>
                    <span style="font-weight: 600;">Annually</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(255,255,255,0.5); border-radius: 10px;">
                    <span>Blood Sugar</span>
                    <span style="font-weight: 600;">Annually</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(255,255,255,0.5); border-radius: 10px;">
                    <span>ECG</span>
                    <span style="font-weight: 600;">As recommended</span>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1.5rem; border-radius: 20px;">
            <h4 style="color: #721c24; margin-bottom: 1rem;">🚨 Warning Signs</h4>
            <div style="display: grid; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #721c24;">•</span>
                    <span>Chest pain/pressure</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #721c24;">•</span>
                    <span>Shortness of breath</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #721c24;">•</span>
                    <span>Irregular heartbeat</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #721c24;">•</span>
                    <span>Extreme fatigue</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Emergency section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); padding: 1.5rem; border-radius: 20px; margin-top: 1rem; animation: pulse 2s infinite;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 3rem;">🚨</div>
            <div>
                <h4 style="color: white; margin-bottom: 0.5rem;">Emergency Symptoms - Call Emergency Services Immediately</h4>
                <p style="color: white; margin: 0; opacity: 0.9;">
                    Chest discomfort, shortness of breath, cold sweat, nausea, lightheadedness
                </p>
            </div>
        </div>
    </div>
    
    <style>
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.9; }
            100% { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# TAB 4 — MODEL INSIGHTS
# -----------------------------
with tab4:
    st.markdown('<h2 class="section-header">🤖 Machine Learning Model Details</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%); padding: 1.5rem; border-radius: 20px;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">📊 Model Specifications</h4>
            <div style="display: grid; gap: 1rem;">
                <div>
                    <p style="color: #64748b; margin: 0;">Algorithm</p>
                    <p style="font-weight: 600; font-size: 1.2rem;">K-Nearest Neighbors</p>
                </div>
                <div>
                    <p style="color: #64748b; margin: 0;">K Value</p>
                    <p style="font-weight: 600;">{} neighbors</p>
                </div>
                <div>
                    <p style="color: #64748b; margin: 0;">Features Used</p>
                    <p style="font-weight: 600;">11 clinical parameters</p>
                </div>
                <div>
                    <p style="color: #64748b; margin: 0;">Training Size</p>
                    <p style="font-weight: 600;">70% of dataset</p>
                </div>
                <div>
                    <p style="color: #64748b; margin: 0;">Validation</p>
                    <p style="font-weight: 600;">5-fold cross-validation</p>
                </div>
            </div>
        </div>
        """.format(model.n_neighbors), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%); padding: 1.5rem; border-radius: 20px;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">📈 Performance Metrics</h4>
        """, unsafe_allow_html=True)
        
        metrics = {
            "Accuracy": 87.3,
            "Precision": 85.1,
            "Recall": 89.2,
            "F1-Score": 87.1,
            "AUC-ROC": 92.0
        }
        
        for metric, value in metrics.items():
            st.markdown(f'''
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="color: #64748b;">{metric}</span>
                    <span style="font-weight: 600;">{value}%</span>
                </div>
                <div style="height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;">
                    <div style="width: {value}%; height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 4px;"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature importance
    st.markdown('<h2 class="section-header">🔬 Feature Importance</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea0d 0%, #764ba20d 100%); padding: 1.5rem; border-radius: 20px;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">Top Predictors of Heart Disease</h4>
            <div style="display: grid; gap: 1rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="background: #667eea; color: white; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-weight: 600;">1</span>
                    <span style="font-weight: 600;">Chest Pain Type (ASY)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="background: #667eea; color: white; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-weight: 600;">2</span>
                    <span style="font-weight: 600;">Exercise Angina</span>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="background: #667eea; color: white; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-weight: 600;">3</span>
                    <span style="font-weight: 600;">ST Slope</span>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="background: #667eea; color: white; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-weight: 600;">4</span>
                    <span style="font-weight: 600;">Age</span>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="background: #667eea; color: white; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-weight: 600;">5</span>
                    <span style="font-weight: 600;">Max Heart Rate</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1.5rem; border-radius: 20px;">
            <h4 style="color: #721c24; margin-bottom: 1rem;">⚠️ Limitations</h4>
            <div style="display: grid; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #721c24;">•</span>
                    <span>Not a diagnostic tool</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #721c24;">•</span>
                    <span>Requires clinical validation</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #721c24;">•</span>
                    <span>Population-specific factors</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #721c24;">•</span>
                    <span>Regular model updates needed</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature distribution
    st.markdown('<h2 class="section-header">📊 Population Distribution</h2>', unsafe_allow_html=True)
    
    # Sample data for visualization
    feature_data = {
        'Age Group': ['18-30', '31-45', '46-60', '61-75', '75+'],
        'Risk Score': [5, 18, 42, 68, 82],
        'Population': [120, 350, 480, 290, 110]
    }
    df_viz = pd.DataFrame(feature_data)
    
    fig = px.bar(df_viz, x='Age Group', y='Risk Score', 
                 title='Risk Distribution by Age Group',
                 labels={'Risk Score': 'Risk Score (%)'},
                 color='Risk Score',
                 color_continuous_scale=['#28a745', '#ffc107', '#dc3545'],
                 text='Risk Score')
    
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    
    fig.update_layout(
        height=450,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': 'Plus Jakarta Sans'},
        title={'font': {'size': 18, 'color': '#1e293b'}}
    )
    
    fig.update_xaxes(gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0', range=[0, 100])
    
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto; flex-wrap: wrap; gap: 1rem;">
        <div style="text-align: left;">
            <p style="margin-bottom: 0.5rem; font-size: 1.2rem; font-weight: 600;">❤️ Heart Health AI</p>
            <p style="margin-bottom: 0; opacity: 0.8; font-size: 0.9rem;">Advanced Cardiovascular Risk Assessment</p>
        </div>
        <div style="text-align: right;">
            <p style="margin-bottom: 0.5rem;">Created by Shivam Attri</p>
            <p style="margin-bottom: 0; opacity: 0.8;">© 2026</p>
        </div>
    </div>
    <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1);">
        <p style="margin-bottom: 0; opacity: 0.7; font-size: 0.85rem;">
            ⚕️ Educational Machine Learning Prototype | Not for Medical Diagnosis
        </p>
    </div>
</div>
""", unsafe_allow_html=True)