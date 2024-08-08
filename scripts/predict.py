from flask import Flask, request, jsonify
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from utils.model_utils import evaluate_model, evaluate_xgboost, get_season
from utils.data_utils import load_data, preprocess_data
from datetime import datetime


def pred(date_str, crop, region, temp, humi, rainfall):
    # Parse the date string in format %dd-%mm-%yyyy
    date = datetime.strptime(date_str, '%d-%m-%Y')
    month = date.month
    year = date.year
    season = get_season(month)

    # Load the data
    df = load_data(fr"C:\Users\USER\Documents\GitHub\Crops\data\filtered_data_{region}_{season}_{crop}.xlsx")
    features = df[['month', 'year', 'temp', 'humi', 'monthly_rainfall_mm', 'region_encoded', 'season_encoded']]
    target = df['amount']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Load the best model
    model_path = f"models/model_{region}_{season}_{crop}.joblib"
    best_model = joblib.load(model_path)

    # Check if the model is XGBoost
    is_xgboost = isinstance(best_model, xgb.Booster)

    # Evaluate the best model
    if is_xgboost:
        xgb_model = best_model
        dtest = xgb.DMatrix(X_test, label=y_test)
        predictions = xgb_model.predict(dtest)
        mae, mse, rmse, r2 = evaluate_xgboost(xgb_model, X_test, y_test)
    else:
        mae, mse, rmse, r2 = evaluate_model(best_model, X_test, y_test)

    evaluation_results = [{
        'Model': 'XGBoost' if is_xgboost else best_model.__class__.__name__,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R^2': r2
    }]

    evaluation_df = pd.DataFrame(evaluation_results)
    print("Evaluation Results:")
    print(evaluation_df)

    evaluation_file_path = fr"C:\Users\USER\Documents\GitHub\Crops\data\evaluation_results_{region}_{season}_{crop}.xlsx"
    evaluation_df.to_excel(evaluation_file_path, index=False)
    print(f"Evaluation results saved to {evaluation_file_path}")

    # Prepare new data for prediction
    new_data = pd.DataFrame({
        'month': [month],
        'year': [year],
        'temp': [temp],
        'humi': [humi],
        'monthly_rainfall_mm': [rainfall],
        'region_encoded': [df['region_encoded'].iloc[0]],  # Replace with actual encoded value if available
        'season_encoded': [df['season_encoded'].iloc[0]]  # Replace with actual encoded value if available
    })

    # Predict with the best model
    if is_xgboost:
        dnew_data = xgb.DMatrix(new_data)
        prediction = xgb_model.predict(dnew_data)[0]
    else:
        prediction = best_model.predict(new_data)[0]

    predictions = {
        'crop': crop,
        'region': region,
        'season': season,
        'amount': prediction
    }

    return predictions, evaluation_results
