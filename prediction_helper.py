import pandas as pd
import os
from joblib import load
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_rest = load(os.path.join(BASE_DIR, "artifacts", "model_rest.joblib"))
model_young = load(os.path.join(BASE_DIR, "artifacts", "model_young.joblib"))

scaler_rest = load(os.path.join(BASE_DIR, "artifacts", "scaler_rest.joblib"))
scaler_young = load(os.path.join(BASE_DIR, "artifacts", "scaler_young.joblib"))

print("All features: ", model_rest.feature_names_in_)

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score

def preprocess_input(input_dict):

    expected_columns = ['age', 'number_of_dependants', 'income_level', 'income_lakhs',
       'insurance_plan', 'genetical_risk',
       'normalized_risk_score', 'gender_Male', 'region_Northwest',
       'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
       'bmi_category_Obesity', 'bmi_category_Overweight',
       'bmi_category_Underweight', 'smoking_status_Occasional',
       'smoking_status_Regular', 'employment_status_Salaried',
       'employment_status_Self-Employed']

    insurance_plan_encoding = {'Bronze': 1, 'Silver':2, 'Gold':3}
    df = pd.DataFrame(0, columns = expected_columns, index=[0])

    for key, value in input_dict.items():

        # -------- NUMERIC -------- #
        if key in ['age', 'number_of_dependants', 'income_lakhs',
                   'annual_premium_amount', 'genetical_risk', 'normalized_risk_score']:
            df[key] = value

        # -------- INSURANCE PLAN -------- #
        elif key == 'insurance_plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 0)

        # -------- GENDER -------- #
        elif key == 'gender':
            if value == 'Male':
                df['gender_Male'] = 1

        # -------- REGION -------- #
        elif key == 'region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1

        # -------- MARITAL STATUS -------- #
        elif key == 'marital_status':
            if value == 'Unmarried':
                df['marital_status_Unmarried'] = 1

        # -------- BMI CATEGORY -------- #
        elif key == 'bmi_category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1

        # -------- SMOKING STATUS -------- #
        elif key == 'smoking_status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1

        # -------- EMPLOYMENT STATUS -------- #
        elif key == 'employment_status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1

        elif key == 'income_level':
            df['income_level'] = income_level_encoding.get(value, 0)

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['medical_history'])
    df = handle_scaling(input_dict['age'], df)
    return df

income_level_encoding = {
            '<10L': 0,
            '10L - 25L': 1,
            '25L - 40L': 2,
            '> 40L': 3
        }

def handle_scaling(age, df):
    if age<=25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df

def predict(input_dict):
    input_dict = {k.lower().replace(" ", "_"): v for k, v in input_dict.items()}
    input_df = preprocess_input(input_dict)

    if input_dict['age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])