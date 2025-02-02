import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import traceback

# Configure the page
st.set_page_config(
    page_title="Transplant Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stAlert {
        padding: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and components with error handling"""
    try:
        model_path = Path(__file__).parent / 'transplant_readmission_model.joblib'
        if not model_path.exists():
            st.error(f"Model file not found. Please ensure 'transplant_readmission_model.joblib' is in the same directory.")
            return None, None, None
        
        components = joblib.load(model_path)
        return components['model'], components['scaler'], components['feature_names']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_gauge_chart(probability):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#00b894'},  # Low risk - green
                {'range': [33, 66], 'color': '#fdcb6e'},  # Medium risk - yellow
                {'range': [66, 100], 'color': '#d63031'}  # High risk - red
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def main():
    try:
        # Load model
        model, scaler, feature_names = load_model()
        if model is None:
            st.stop()
        
        # Page Header
        st.title("🏥 Transplant Readmission Risk Predictor")
        st.markdown("""
        This tool predicts the risk of hospital readmission for transplant patients based on clinical parameters.
        Please enter the patient's information below for risk assessment.
        """)
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            
            # Basic Information
            gender = st.selectbox("Gender", ["Male", "Female"])
            blood_group = st.selectbox("Blood Group", ["A", "B", "AB", "O"])
            transplant_type = st.selectbox("Transplant Type", ["Living", "Deceased"])
            diabetes = st.selectbox("Diabetes Status", ["Yes", "No"])
            
            # Pre-transplant Measurements
            st.subheader("Pre-transplant Measurements")
            pre_bmi = st.number_input("Pre-transplant BMI (kg/m²)", 15.0, 50.0, 25.0, help="Body Mass Index before transplantation")
            pre_hba1c = st.number_input("Pre-transplant HbA1c (%)", 4.0, 15.0, 5.7, help="Glycated hemoglobin before transplantation")
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
        
        # Predict button
        if st.button("Predict Readmission Risk", use_container_width=True):
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
            
            # Display results
            st.markdown("---")
            st.subheader("Risk Assessment Results")
            
            # Create three columns for results
            res_col1, res_col2, res_col3 = st.columns([2,1,1])
            
            with res_col1:
                # Display gauge chart
                fig = create_gauge_chart(risk_prob)
                st.plotly_chart(fig)
            
            with res_col2:
                st.markdown(f"""
                ### Risk Level
                **{'🔴 High' if risk_prob > 0.66 else '🟡 Medium' if risk_prob > 0.33 else '🟢 Low'} Risk**
                
                Probability: {risk_prob:.1%}
                
                Confidence: {max(risk_prob, 1-risk_prob):.1%}
                """)
            
            with res_col3:
                st.markdown("### Recommended Actions")
                if risk_prob > 0.66:
                    st.markdown("""
                    - 📋 Close monitoring required
                    - 🏥 Early follow-up visit (within 48h)
                    - 📞 Daily check-ins
                    - 🔍 Intensive monitoring
                    """)
                elif risk_prob > 0.33:
                    st.markdown("""
                    - 📋 Regular monitoring
                    - 🏥 Follow-up within 1 week
                    - ⚕️ Review medications
                    - 📞 Twice-weekly check-ins
                    """)
                else:
                    st.markdown("""
                    - 📋 Routine monitoring
                    - 🏥 Regular schedule
                    - ✅ Standard care
                    - 📞 Weekly check-ins
                    """)
            
            # Display key risk factors if high risk
            if risk_prob > 0.5:
                st.markdown("---")
                st.subheader("Key Risk Factors")
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(5)
                
                st.write("Top 5 factors influencing this prediction:")
                for idx, row in feature_importance.iterrows():
                    st.write(f"- {row['Feature']}: {row['Importance']:.3f}")
        
        # Add footer with information
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        <p>This tool is intended to assist clinical decision-making and should not replace professional medical judgment.</p>
        <p>For emergency situations, please contact your healthcare provider immediately.</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error("An error occurred in the application.")
        st.error(f"Error details: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
