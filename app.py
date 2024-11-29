import streamlit as st
import pandas as pd
from joblib import load

def main():

    # Load the trained models
    recovery_time_model = load('models/trained_recovery_time_model.joblib')

    # Title and description of the app
    st.title("Knee Recovery Prediction")
    st.write("This application predicts the recovery time based on the patient's knee condition.")

    # User input for prediction
    patient_type = st.selectbox("Patient Type", ["Normal", "Mild", "Moderate", "Severe"])
    flexion_angle = st.number_input("Flexion Angle (degrees)", min_value=0, max_value=180, step=1)
    flexion_category = st.selectbox("Flexion Category", ["fully_stretched", "partially_bent", "fully_bent"])

    # Create a base dictionary of features
    input_data = {
        'Patient_Type_Normal': 1 if patient_type == 'Normal' else 0,
        'Patient_Type_Mild': 1 if patient_type == 'Mild' else 0,
        'Patient_Type_Moderate': 1 if patient_type == 'Moderate' else 0,
        'Patient_Type_Severe': 1 if patient_type == 'Severe' else 0,
        'Flexion_Angle': flexion_angle,
        'Flexion_Category_partially_bent': 1 if flexion_category == 'partially_bent' else 0,
        'Flexion_Category_fully_bent': 1 if flexion_category == 'fully_bent' else 0,
        'Flexion_Category_fully_stretched': 1 if flexion_category == 'fully_stretched' else 0
    }

    # Convert the dictionary to DataFrame
    input_data_df = pd.DataFrame(input_data, index=[0])

    # Get the column names that the model expects
    expected_columns = recovery_time_model.feature_names_in_  # Get feature names from the model

    # Add missing columns (if any) with 0 values
    missing_columns = [col for col in expected_columns if col not in input_data_df.columns]
    for col in missing_columns:
        input_data_df[col] = 0

    # Reorder columns to match the expected order during training
    input_data_df = input_data_df[expected_columns]

    # Store prediction results in session state to retain across button clicks
    if 'recovery_time_output' not in st.session_state:
        st.session_state.recovery_time_output = None

    # Button for predicting recovery time
    if st.button("Predict Recovery Time (Weeks)"):
        recovery_time_prediction = recovery_time_model.predict(input_data_df)
        # Round the prediction to the nearest whole number
        st.session_state.recovery_time_output = f"Predicted Recovery Time: {round(recovery_time_prediction[0])} weeks"

    # Display the results, if any, in big letters
    if st.session_state.recovery_time_output:
        st.markdown(f"### {st.session_state.recovery_time_output.upper()}")

if __name__ == "__main__":
    main()
