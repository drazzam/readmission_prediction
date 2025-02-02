import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import traceback
from clinical_validator import ClinicalValidator

# Initialize clinical validator
validator = ClinicalValidator()

# Page config with custom styling
st.set_page_config(
    page_title="Clinical Transplant Readmission Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .clinical-alert {
        background-color: #ffe4e4;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model with enhanced error handling"""
    try:
        model_path = Path(__file__).parent / 'transplant_readmission_model.joblib'
        if not model_path.exists():
            st.error("Model file not found. Please verify model deployment.")
            return None, None, None
        
        components = joblib.load(model_path)
        return components['model'], components['scaler'], components['feature_names']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_clinical_visualization(risk_metrics, clinical_values):
    """Create comprehensive clinical visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "bar", "colspan": 2}, None]],
        subplot_titles=("Risk Score", "Clinical Metrics", "Key Risk Factors")
    )
    
    # Risk gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_metrics['clinical_probability'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 67], 'color': "yellow"},
                    {'range': [67, 100], 'color': "pink"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_metrics['clinical_probability'] * 100
                }
            }
        ),
        row=1, col=1
    )
    
    # Clinical metrics
    for idx, (metric, value) in enumerate(clinical_values.items()):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=[metric],
                y=[value],
                text=[f"{value:.1f}"],
                textposition='auto',
            ),
            row=2, col=1
        )
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def main():
    st.title("🏥 Clinical Transplant Readmission Risk Predictor")
    st.markdown("""
        This tool provides evidence-based risk assessment for post-transplant readmission.
        Model performance: AUC-ROC: 0.837 (95% CI: 0.802-0.872)
    """)
    
    # Load model
    model, scaler, feature_names = load_model()
    if model is None:
        st.stop()
    
    # Input form with clinical context
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Baseline Characteristics")
        input_data = {}
        
        # Demographic inputs
        input_data['gender'] = st.selectbox("Gender", ["Male", "Female"])
        input_data['blood_group'] = st.selectbox("Blood Group", ["A", "B", "AB", "O"])
        input_data['transplant_type'] = st.selectbox("Transplant Type", ["Living", "Deceased"])
        input_data['diabetes'] = st.selectbox("Diabetes Status", ["Yes", "No"])
        
        # Pre-transplant measurements
        st.subheader("Pre-transplant Clinical Parameters")
        input_data['pre_bmi'] = st.number_input("Pre-transplant BMI (kg/m²)", 15.0, 50.0, 25.0)
        input_data['pre_hba1c'] = st.number_input("Pre-transplant HbA1c (%)", 4.0, 15.0, 5.7)
        input_data['pre_systolic'] = st.number_input("Pre-transplant Systolic BP (mmHg)", 80, 200, 120)
        input_data['pre_diastolic'] = st.number_input("Pre-transplant Diastolic BP (mmHg)", 40, 120, 80)
        input_data['pre_egfr'] = st.number_input("Pre-transplant eGFR (mL/min)", 0.0, 120.0, 60.0)
    
    with col2:
        st.subheader("Post-transplant Parameters")
        input_data['post_bmi'] = st.number_input("Post-transplant BMI (kg/m²)", 15.0, 50.0, 25.0)
        input_data['post_hba1c'] = st.number_input("Post-transplant HbA1c (%)", 4.0, 15.0, 5.7)
        input_data['post_systolic'] = st.number_input("Post-transplant Systolic BP (mmHg)", 80, 200, 120)
        input_data['post_diastolic'] = st.number_input("Post-transplant Diastolic BP (mmHg)", 40, 120, 80)
        input_data['post_egfr'] = st.number_input("Post-transplant eGFR (mL/min)", 0.0, 120.0, 60.0)
        input_data['post_creatinine'] = st.number_input("Post-transplant Creatinine (mg/dL)", 0.0, 10.0, 1.2)
        input_data['length_of_stay'] = st.number_input("Length of Stay (days)", 1, 60, 7)
        input_data['immunosuppression'] = st.selectbox("Immunosuppression Regimen", ["Standard", "Modified"])

    # Risk Assessment
    if st.button("Generate Clinical Risk Assessment", use_container_width=True):
        try:
            # Prepare input data
            X = prepare_input_data(input_data)
            X_scaled = scaler.transform(X)
            
            # Get model prediction
            base_prob = model.predict_proba(X_scaled)[0][1]
            
            # Get clinical validation and risk metrics
            risk_metrics = validator.calculate_risk_metrics(base_prob, input_data)
            clinical_interpretation = validator.get_clinical_interpretation(
                risk_metrics['risk_level'],
                input_data
            )
            
            # Display results
            st.markdown("---")
            st.subheader("Clinical Risk Assessment Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_clinical_visualization(risk_metrics, input_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Clinical Interpretation")
                st.markdown(f"""
                    **Risk Level:** {risk_metrics['risk_level'].replace('_', ' ').title()}
                    
                    **Probability:** {risk_metrics['clinical_probability']:.1%}
                    (95% CI: {risk_metrics['confidence_interval'][0]:.1%} - {risk_metrics['confidence_interval'][1]:.1%})
                    
                    {clinical_interpretation['recommendation']}
                """)
                
                if 'alert' in clinical_interpretation:
                    st.markdown(f"""
                        <div class='clinical-alert'>
                            ⚠️ {clinical_interpretation['alert']}
                        </div>
                    """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.markdown("### Detailed Clinical Metrics")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4>
