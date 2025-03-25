import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
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
    .donor-specific {
        border-left: 4px solid #1E88E5;
        padding-left: 10px;
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .rejection-risk-high {
        border-left: 4px solid #D32F2F;
        background-color: #FFEBEE;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .rejection-risk-low {
        border-left: 4px solid #388E3C;
        background-color: #E8F5E9;
        padding: 10px;
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

def create_feature_importance_chart(donor_type, input_data, feature_importances):
    """Create a horizontal bar chart of feature importances based on donor type"""
    # Select the appropriate feature importance values based on donor type
    if donor_type == "Living":
        importances = feature_importances["living"]
    else:
        importances = feature_importances["deceased"]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Feature': list(importances.keys()),
        'Importance (%)': list(importances.values())
    })
    
    # Sort by importance
    df = df.sort_values('Importance (%)', ascending=True)
    
    # Create figure
    fig = px.bar(
        df.tail(7),  # Take the top 7 features
        y='Feature', 
        x='Importance (%)',
        orientation='h',
        title=f'Key Predictors for {donor_type} Donor Recipients',
        color='Importance (%)',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    
    return fig

def get_risk_category(probability):
    if probability > 0.67:
        return "High", "🔴"
    elif probability > 0.33:
        return "Medium", "🟡"
    else:
        return "Low", "🟢"

def calculate_rejection_risk(donor_type, input_data):
    """Calculate graft rejection risk based on donor type and clinical parameters"""
    base_risk = 0.138 if donor_type == "Living" else 0.471  # Base rates from our analysis
    
    # Adjust risk based on key factors
    risk_adjustments = 0
    
    # Pre-transplant BMI (more important for deceased donors)
    if donor_type == "Deceased" and input_data["PreBMI"] > 30:
        risk_adjustments += 0.05
    
    # eGFR (more important for living donors)
    if input_data["PosteGFR"] < 30:
        adjustment = 0.08 if donor_type == "Living" else 0.03
        risk_adjustments += adjustment
    
    # HbA1c (more important for living donors)
    if input_data["PreHbA1c"] > 6.5:
        adjustment = 0.06 if donor_type == "Living" else 0.02
        risk_adjustments += adjustment
    
    # Length of stay (important for both)
    if input_data["LengthOfStay"] > 7:
        risk_adjustments += 0.05
    
    # Adjust base risk with our calculated adjustments
    adjusted_risk = min(0.95, max(0.05, base_risk + risk_adjustments))
    
    return adjusted_risk

def adjust_prediction_by_donor(base_prediction, donor_type, input_data):
    """Adjust the prediction probability based on donor type and key parameters"""
    # Start with the base model prediction
    adjusted_prediction = base_prediction
    
    # Apply donor-specific adjustments
    if donor_type == "Deceased":
        # Adjust for Pre-BMI (more important for deceased donors)
        if input_data["PreBMI"] > 30:
            adjusted_prediction = min(0.95, adjusted_prediction * 1.15)
            
        # Higher baseline readmission rate for deceased donors
        adjusted_prediction = min(0.95, adjusted_prediction * 1.05)
        
    else:  # Living donor
        # Adjust for HbA1c (more important for living donors)
        if input_data["PreHbA1c"] > 6.5:
            adjusted_prediction = min(0.95, adjusted_prediction * 1.10)
            
        # Adjust for eGFR (more important for living donors)
        if input_data["PosteGFR"] < 30:
            adjusted_prediction = min(0.95, adjusted_prediction * 1.08)
    
    return adjusted_prediction

def main():
    # Load model
    model, scaler, feature_names, encoders = load_model()
    if model is None:
        st.stop()
    
    # Define donor-specific feature importances based on our analysis
    feature_importances = {
        "living": {
            "Length of hospital stay": 24.5,
            "Post-transplant systolic BP": 21.2,
            "Post-transplant eGFR": 8.8,
            "Pre-transplant HbA1c": 9.9,
            "Post-transplant BMI": 5.4,
            "Pre-transplant BMI": 2.6,
            "Pre-transplant diastolic BP": 1.5
        },
        "deceased": {
            "Length of hospital stay": 20.1,
            "Post-transplant systolic BP": 16.7,
            "Pre-transplant BMI": 12.6,
            "Pre-transplant HbA1c": 3.5,
            "Post-transplant eGFR": 2.6,
            "Post-transplant BMI": 3.3,
            "Pre-transplant diastolic BP": 2.9
        }
    }
    
    # Page Header
    st.title("🏥 Transplant Readmission Risk Predictor")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_group = st.selectbox("Blood Group", ["A", "B", "AB", "O"])
        transplant_type = st.selectbox("Transplant Type", ["Living", "Deceased"])
        
        # Show donor-specific model performance
        if transplant_type == "Living":
            st.markdown("""
            <div class="donor-specific">
            <b>Living Donor Model Performance:</b><br>
            AUC = 0.736 (95% CI: 0.576-0.897)<br>
            Sensitivity = 0.79, Specificity = 0.69
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="donor-specific">
            <b>Deceased Donor Model Performance:</b><br>
            AUC = 0.708 (95% CI: 0.491-0.925)<br>
            Sensitivity = 0.81, Specificity = 0.71
            </div>
            """, unsafe_allow_html=True)
        
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
            
            # Make base prediction
            base_risk_prob = model.predict_proba(input_scaled)[0][1]
            
            # Adjust prediction based on donor type
            risk_prob = adjust_prediction_by_donor(base_risk_prob, transplant_type, input_data)
            
            # Calculate graft rejection risk
            rejection_risk = calculate_rejection_risk(transplant_type, input_data)
            
            # Get readmission risk category
            risk_category, risk_emoji = get_risk_category(risk_prob)
            
            # Display results
            st.markdown("---")
            st.subheader("Risk Assessment Results")
            
            risk_cols = st.columns([2,1,1])
            
            with risk_cols[0]:
                fig = create_gauge_chart(risk_prob)
                st.plotly_chart(fig)
            
            with risk_cols[1]:
                st.markdown(f"""
                ### Readmission Risk
                **{risk_emoji} {risk_category} Risk**
                
                Probability: {risk_prob:.1%}
                
                Donor Type: {transplant_type}
                """)
                
                # Display graft rejection risk 
                if transplant_type == "Deceased":
                    st.markdown(f"""
                    <div class="rejection-risk-high">
                    <b>⚠️ Graft Rejection Risk: {rejection_risk:.1%}</b><br>
                    Deceased donor recipients have significantly higher rejection rates (47.1% vs 13.8%)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="rejection-risk-low">
                    <b>✓ Graft Rejection Risk: {rejection_risk:.1%}</b><br>
                    Living donor recipients have lower rejection rates (13.8% vs 47.1%)
                    </div>
                    """, unsafe_allow_html=True)
            
            with risk_cols[2]:
                st.markdown("### Recommended Actions")
                if risk_prob > 0.67:
                    st.markdown("""
                    - 📋 Close monitoring required
                    - 🏥 Early follow-up (48h)
                    - 📞 Daily check-ins
                    - 🔍 Intensive monitoring
                    """)
                    if transplant_type == "Deceased":
                        st.markdown("- 🔬 Monitor for rejection signs")
                elif risk_prob > 0.33:
                    st.markdown("""
                    - 📋 Regular monitoring
                    - 🏥 Follow-up within 1 week
                    - 📞 Twice-weekly check-ins
                    - ⚕️ Review medications
                    """)
                    if transplant_type == "Deceased":
                        st.markdown("- 🩸 Consider more frequent labs")
                else:
                    st.markdown("""
                    - 📋 Routine monitoring
                    - 🏥 Standard schedule
                    - 📞 Weekly check-ins
                    - ✅ Standard care
                    """)
            
            # Add feature importance visualization
            st.markdown("---")
            st.subheader("Key Risk Predictors")
            importance_chart = create_feature_importance_chart(transplant_type, input_data, feature_importances)
            st.plotly_chart(importance_chart, use_container_width=True)
            
            # Clinical Interpretation
            st.markdown("---")
            st.subheader("Clinical Interpretation")
            
            interpretation_cols = st.columns(3)
            with interpretation_cols[0]:
                st.markdown(f"""
                **Key Risk Factors:**
                - Length of Stay: {length_of_stay} days
                - Post-transplant Systolic BP: {post_systolic} mmHg
                - {'Pre-transplant BMI: ' + str(pre_bmi) + ' kg/m²' if transplant_type == 'Deceased' else 'Pre-transplant HbA1c: ' + str(pre_hba1c) + '%'}
                - {'Post-transplant eGFR: ' + str(post_egfr) + ' mL/min' if transplant_type == 'Living' else 'Post-transplant Creatinine: ' + str(post_creatinine) + ' mg/dL'}
                """)
            
            with interpretation_cols[1]:
                st.markdown(f"""
                **Donor-Specific Context:**
                - {transplant_type} donor transplant ({('47.1% rejection risk' if transplant_type == 'Deceased' else '13.8% rejection risk')})
                - {'⚠️ BMI is a stronger predictor for deceased donors' if transplant_type == 'Deceased' else '⚠️ HbA1c and eGFR are stronger predictors for living donors'}
                - {'⚠️ Extended stay' if length_of_stay > 7 else '✓ Normal stay'}
                - {'⚠️ High BP' if post_systolic > 140 else '✓ Controlled BP'}
                """)
            
            with interpretation_cols[2]:
                st.markdown(f"""
                **Model Confidence:**
                - Prediction confidence: {max(risk_prob, 1-risk_prob):.1%}
                - Based on {length_of_stay} days of monitoring
                - {transplant_type} donor model: AUC {'0.736' if transplant_type == 'Living' else '0.708'}
                - {'⚠️ Wider confidence intervals for deceased donor model' if transplant_type == 'Deceased' else ''}
                """)
        
        except Exception as e:
            st.error("An error occurred during prediction.")
            st.error(f"Error details: {str(e)}")
            st.error(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
        <p>This tool is intended to assist clinical decision-making and should not replace professional medical judgment.</p>
        <p>Model performance varies by donor type (Living: AUC 0.736, Deceased: AUC 0.708).</p>
        <p>For emergency situations, please contact your healthcare provider immediately.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
