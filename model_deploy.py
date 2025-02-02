import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import traceback

# Page config
st.set_page_config(
    page_title="Transplant Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 20px;}
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
    try:
        model_path = Path('transplant_readmission_model.joblib')
        if not model_path.exists():
            st.error("Model file not found. Please ensure the model file is present.")
            return None, None, None, None
        
        components = joblib.load(model_path)
        return (components['model'], components['scaler'], 
                components['feature_names'], components['categorical_encoders'])
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#00b894'},
                {'range': [33, 67], 'color': '#fdcb6e'},
                {'range': [67, 100], 'color': '#d63031'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def get_risk_category(probability):
    if probability > 0.67:
        return "High", "🔴"
    elif probability > 0.33:
        return "Medium", "🟡"
    else:
        return "Low", "🟢"

def main():
    # Load model
    model, scaler, feature_names, encoders = load_model()
    if model is None:
        st.stop()
    
    # Page Header
    st.title("🏥 Transplant Readmission Risk Predictor")
    st.markdown("""
    This tool predicts the risk of hospital readmission for transplant patients based on clinical parameters.
    Model Performance: AUC-ROC = 0.837 (95% CI: 0.802-0.872)
    """)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_group = st.selectbox("Blood Group", ["A", "B", "AB", "O"])
        transplant_type = st.selectbox("Transplant Type", ["Living", "Deceased"])
        diabetes = st.selectbox("Diabetes Status", ["Yes", "No"])
        
        st.subheader("Pre-transplant Measurements")
        pre_bmi = st.number_input("Pre-transplant BMI (kg/m²)", 15.0, 50.0, 25.0)
        pre_hba1c = st.number_input("Pre-transplant HbA1c (%)", 4.0, 15.0, 5.7)
        pre_systolic = st.number_input("Pre-transplant Systolic BP (mmHg)", 80, 200, 120)
        pre_diastolic = st.number_input("Pre-transplant Diastolic BP (mmHg)", 40, 120, 80)
        pre_egfr = st.number_input("Pre-transplant eGFR (mL/min)", 0.0, 120.0, 60.0)
    
    with col2:
        st.subheader("Post-transplant Parameters")
        post_bmi = st.number_input("Post-transplant BMI (kg/m²)", 15.0, 50.0, 25.0)
        post_hba1c = st.number_input("Post-transplant HbA1c (%)", 4.0, 15.0, 5.7)
        post_systolic = st.number_input("Post-transplant Systolic BP (mmHg)", 80, 200, 120)
        post_diastolic = st.number_input("Post-transplant Diastolic BP (mmHg)", 40, 120, 80)
        post_egfr = st.number_input("Post-transplant eGFR (mL/min)", 0.0, 120.0, 60.0)
        post_creatinine = st.number_input("Post-transplant Creatinine (mg/dL)", 0.0, 10.0, 1.2)
        length_of_stay = st.number_input("Length of Stay (days)", 1, 60, 7)
        immunosuppression = st.selectbox("Immunosuppression Regimen", ["Standard", "Modified"])
    
    if st.button("Predict Readmission Risk", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'Gender': 1 if gender == "Male" else 0,
                'BloodGroup': ["A", "B", "AB", "O"].index(blood_group),
                'TransplantType': 1 if transplant_type == "Living" else 0,
                'PreBMI': pre_bmi,
                'PreHbA1c': pre_hba1c,
                'PreSystolicBP': pre_systolic,
                'PreDiastolicBP': pre_diastolic,
                'PreeGFR': pre_egfr,
                'PostBMI': post_bmi,
                'PostHbA1c': post_hba1c,
                'PostsystolicBP': post_systolic,
                'PostdiastolicBP': post_diastolic,
                'PosteGFR': post_egfr,
                'PostCreatinine': post_creatinine,
                'LengthOfStay': length_of_stay,
                'ImmunosuppressiononDischarge': 1 if immunosuppression == "Standard" else 0,
                'diabetes': 1 if diabetes == "Yes" else 0
            }
            
            # Convert to DataFrame and scale
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            risk_prob = model.predict_proba(input_scaled)[0][1]
            risk_category, risk_emoji = get_risk_category(risk_prob)
            
            # Display results
            st.markdown("---")
            st.subheader("Risk Assessment Results")
            
            col1, col2, col3 = st.columns([2,1,1])
            
            with col1:
                fig = create_gauge_chart(risk_prob)
                st.plotly_chart(fig)
            
            with col2:
                st.markdown(f"""
                ### Risk Level
                **{risk_emoji} {risk_category} Risk**
                
                Probability: {risk_prob:.1%}
                """)
            
            with col3:
                st.markdown("### Recommended Actions")
                if risk_prob > 0.67:
                    st.markdown("""
                    - 📋 Close monitoring required
                    - 🏥 Early follow-up (48h)
                    - 📞 Daily check-ins
                    - 🔍 Intensive monitoring
                    """)
                elif risk_prob > 0.33:
                    st.markdown("""
                    - 📋 Regular monitoring
                    - 🏥 Follow-up within 1 week
                    - 📞 Twice-weekly check-ins
                    - ⚕️ Review medications
                    """)
                else:
                    st.markdown("""
                    - 📋 Routine monitoring
                    - 🏥 Standard schedule
                    - 📞 Weekly check-ins
                    - ✅ Standard care
                    """)
            
            # Clinical Interpretation
            st.markdown("---")
            st.subheader("Clinical Interpretation")
            
            interpretation_cols = st.columns(3)
            with interpretation_cols[0]:
                st.markdown(f"""
                **Key Risk Factors:**
                - Length of Stay: {length_of_stay} days
                - Creatinine: {post_creatinine} mg/dL
                - Systolic BP: {post_systolic} mmHg
                """)
            
            with interpretation_cols[1]:
                st.markdown(f"""
                **Clinical Context:**
                - {'⚠️ Extended stay' if length_of_stay > 7 else '✓ Normal stay'}
                - {'⚠️ Elevated creatinine' if post_creatinine > 2.0 else '✓ Normal creatinine'}
                - {'⚠️ High BP' if post_systolic > 140 else '✓ Controlled BP'}
                """)
            
            with interpretation_cols[2]:
                st.markdown(f"""
                **Model Confidence:**
                - Prediction confidence: {max(risk_prob, 1-risk_prob):.1%}
                - Based on {length_of_stay} days of monitoring
                - Clinical validation: AUC 0.837
                """)
        
        except Exception as e:
            st.error("An error occurred during prediction.")
            st.error(f"Error details: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
        <p>This tool is intended to assist clinical decision-making and should not replace professional medical judgment.</p>
        <p>For emergency situations, please contact your healthcare provider immediately.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
