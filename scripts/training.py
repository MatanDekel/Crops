#function that get the data from the filter
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from project.utils.model_utils import train_models, train_xgboost, evaluate_model, evaluate_xgboost
import joblib


def train(df, region, season, crop):
    if df.empty:
        return
    if len(df) <= 50:
        return
    features = df[['month', 'year', 'temp', 'humi', 'monthly_rainfall_mm', 'region_encoded', 'season_encoded']]
    target = df['amount']

    if features.empty or target.empty:
        raise ValueError(
            "Features or target data is empty after selecting columns. Please check the DataFrame contents.")
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    train_models(X_train, y_train)

    xgb_model = train_xgboost(X_train, y_train)

    evaluation_results = []

    for model_name in ['linear_regression', 'decision_tree', 'random_forest']:
        model = joblib.load(f"models/{model_name}.joblib")
        mae, mse, rmse, r2 = evaluate_model(model, X_test, y_test)
        evaluation_results.append({
            'Model': model_name.replace('_', ' ').title(),
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R^2': r2
        })

    xgb_mae, xgb_mse, xgb_rmse, xgb_r2 = evaluate_xgboost(xgb_model, X_test, y_test)
    evaluation_results.append({
        'Model': 'XGBoost',
        'MAE': xgb_mae,
        'MSE': xgb_mse,
        'RMSE': xgb_rmse,
        'R^2': xgb_r2
    })

    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.to_excel(fr"D:\Final project\pythonProject2\project\data\evaluation_results_{region}_{season}_{crop}.xlsx",
                           index=False)
    print(f"Evaluation results saved to evaluation_results_{region}_{season}_{crop}.xlsx")
    print(evaluation_df)
