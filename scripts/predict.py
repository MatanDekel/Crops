from flask import Flask, request, jsonify
import pandas as pd
import joblib
import xgboost as xgb
from utils.model_utils import get_season
from utils.data_utils import load_data
from datetime import datetime


def pred(date_str, crop, region):
    if not isinstance(date_str, str):
        raise ValueError("The date must be a string in the format 'dd-mm-yyyy'.")
    if not isinstance(crop, str):
        raise ValueError("The crop must be a string.")
    if not isinstance(region, str):
        raise ValueError("The region must be a string.")

    try:
        date = datetime.strptime(date_str, '%d-%m-%Y')
    except ValueError:
        raise ValueError("The date must be in the format 'dd-mm-yyyy'.")

    month = int(date.month)
    season = get_season(month)

    # Load the data
    data_path = f"data/filtered_data/filtered_data_{region}_{season}_{crop}.xlsx"
    df = load_data(data_path)
    required_columns = ['month', 'year', 'temp', 'humi', 'monthly_rainfall_mm', 'region_encoded', 'season_encoded']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter the DataFrame to include only the features used during training
    df = df[required_columns]

    # Convert columns to the correct types
    df['month'] = pd.to_numeric(df['month'], errors='coerce').astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce').astype(float)
    df['humi'] = pd.to_numeric(df['humi'], errors='coerce').astype(float)
    df['monthly_rainfall_mm'] = pd.to_numeric(df['monthly_rainfall_mm'], errors='coerce').astype(float)
    df['region_encoded'] = pd.to_numeric(df['region_encoded'], errors='coerce').astype(int)
    df['season_encoded'] = pd.to_numeric(df['season_encoded'], errors='coerce').astype(int)

    # Load the best model
    model_path = f"scripts/models/model_{region}_{season}_{crop}.joblib"
    try:
        xgb_model = xgb.Booster()
        xgb_model.load_model(model_path)
        is_xgboost = True
    except xgb.core.XGBoostError:
        best_model = joblib.load(model_path)
        is_xgboost = False

    # Predict with the best model
    if is_xgboost:
        dnew_data = xgb.DMatrix(df)
        prediction = xgb_model.predict(dnew_data)[0]
    else:
        prediction = best_model.predict(df)[0]

    predictions = {
        'crop': crop,
        'region': region,
        'season': season,
        'amount': prediction
    }

    return predictions
