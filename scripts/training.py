import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.model_utils import train_models, train_xgboost, evaluate_model, evaluate_xgboost, best_model, save_model
import joblib
import xgboost as xgb

our_crops = ['Cucumbers', 'Peppers', 'Tomatoes', 'Clementines', 'Potatoes']


def train(df, region, season, crop):
    # Check if the dataframe is empty or has insufficient data
    print(df)
    if crop in our_crops:
        # Extract features and target variables
        features = df[['month', 'year', 'temp', 'humi', 'monthly_rainfall_mm', 'region_encoded', 'season_encoded']]
        target = df['amount']

        # Ensure features and target are not empty
        if features.empty or target.empty:
            raise ValueError(
                "Features or target data is empty after selecting columns. Please check the DataFrame contents.")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train various models
        models = train_models(X_train, y_train)

        # Evaluate the models
        evaluation_results = []
        for model_name, model in models.items():
            mae, mse, rmse, r2 = evaluate_model(model, X_test, y_test)
            evaluation_results.append({
                'Model': model_name.replace('_', ' ').title(),
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R^2': r2
            })

        # Train and evaluate the XGBoost model using xgb.train
        xgb_model = train_xgboost(X_train, y_train)
        xgb_mae, xgb_mse, xgb_rmse, xgb_r2 = evaluate_xgboost(xgb_model, X_test, y_test)
        evaluation_results.append({
            'Model': 'XGBoost',
            'MAE': xgb_mae,
            'MSE': xgb_mse,
            'RMSE': xgb_rmse,
            'R^2': xgb_r2
        })

        # Save the evaluation results to an Excel file
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_file_path = fr"../data/evaluation/evaluation_{region}_{season}_{crop}.xlsx"
        evaluation_df.to_excel(evaluation_file_path, index=False)
        print(f"Evaluation results saved to {evaluation_file_path}")
        print(evaluation_df)

        # Identify and save the best model
        best_models = best_model(evaluation_file_path)
        if best_models:
            best_model_name = best_models[0].replace(' ', '_').lower()
            best_model_instance = models.get(best_model_name, xgb_model if best_model_name == 'xgboost' else None)
            if best_model_instance:
                if best_model_name == 'xgboost':
                    # Save the XGBoost model using xgb.Booster.save_model()
                    model_save_path = f"models/model_{region}_{season}_{crop}.joblib"
                    xgb_model.save_model(model_save_path)
                    print(f"XGBoost model saved to {model_save_path}")
                else:
                    # Save other models using joblib
                    save_model(best_model_instance, best_model_name, region, season, crop)
