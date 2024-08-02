import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


# Load models and label encoders
def load_models():
    models = {}
    for model_name in ['linear_regression', 'decision_tree', 'random_forest', 'xgboost']:
        model_path = f"models/{model_name}.joblib"
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
        else:
            print(f"Model {model_name} not found at {model_path}. Skipping...")
    return models


models = load_models()

# Assuming the filtered DataFrame is already loaded and preprocessed
filtered_df = pd.read_excel('filtered_data.xlsx')
le_region = LabelEncoder().fit(filtered_df['region'])
le_season = LabelEncoder().fit(filtered_df['season'])


def preprocess_input_data(df, date, le_region, le_season):
    try:
        date = pd.to_datetime(date, format='%d-%m-%Y')
    except ValueError:
        raise ValueError("Date should be in 'DD-MM-YYYY' format")

    month = date.month
    year = date.year

    input_row = df.iloc[0]
    temp = input_row['temp']
    humi = input_row['humi']
    monthly_rainfall_mm = input_row['monthly_rainfall_mm']
    region_encoded = input_row['region_encoded']
    season_encoded = input_row['season_encoded']

    input_df = pd.DataFrame({
        'month': [month],
        'year': [year],
        'temp': [temp],
        'humi': [humi],
        'monthly_rainfall_mm': [monthly_rainfall_mm],
        'region_encoded': [region_encoded],
        'season_encoded': [season_encoded]
    })

    return input_df


def predict_crop_amount(models, input_df):
    predictions = {}
    for model_name, model in models.items():
        if model_name == 'xgboost':
            predictions[model_name] = model.predict(input_df)[0]
        else:
            predictions[model_name] = model.predict(input_df)[0]
    return predictions


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    crop = data.get('crop')
    date = data.get('date')

    if not crop or not date:
        return jsonify({'error': 'Please provide crop and date'}), 400

    input_df = preprocess_input_data(filtered_df, date, le_region, le_season)
    predictions = predict_crop_amount(models, input_df)

    return jsonify(predictions)
