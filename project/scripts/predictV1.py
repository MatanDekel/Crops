import pandas as pd
import joblib
import xgboost as xgb
from project.utils.model_utils import evaluate_model, evaluate_xgboost

if __name__ == "__main__":
    preprocessed_file_path = r"D:\Final project\pythonProject2\data\preprocessed_data.csv"
    df = pd.read_csv(preprocessed_file_path)

    # Select features and target variable
    features = df[['month', 'year', 'temp', 'humi', 'monthly_rainfall_mm']]
    target = df['amount']  # Replace 'amount' with your actual target column name

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Load models
    models = {
        'Linear Regression': joblib.load("models/linear_regression.joblib"),
        'Decision Tree': joblib.load("models/decision_tree.joblib"),
        'Random Forest': joblib.load("models/random_forest.joblib"),
    }
    xgb_model = xgb.Booster()
    xgb_model.load_model("models/xgboost_model.json")

    # Evaluate models
    evaluation_results = []
    for name, model in models.items():
        mae, mse, rmse, r2 = evaluate_model(model, X_test, y_test)
        evaluation_results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R^2': r2
        })

    # Evaluate XGBoost model
    xgb_mae, xgb_mse, xgb_rmse, xgb_r2 = evaluate_xgboost(xgb_model, X_test, y_test)
    evaluation_results.append({
        'Model': 'XGBoost',
        'MAE': xgb_mae,
        'MSE': xgb_mse,
        'RMSE': xgb_rmse,
        'R^2': xgb_r2
    })

    # Print evaluation results
    evaluation_df = pd.DataFrame(evaluation_results)
    print("Evaluation Results:")
    print(evaluation_df)

    # Save evaluation results to a CSV file
    evaluation_df.to_csv(r"D:\Final project\pythonProject2\data\evaluation_results.csv", index=False)
    print("Evaluation results saved to evaluation_results.csv")

    # Make predictions (if needed)
    new_data = pd.DataFrame({
        'month': [4],  # Example month
        'year': [2023],  # Example year
        'temp': [20],  # Example temperature
        'humi': [60],  # Example humidity
        'monthly_rainfall_mm': [100]  # Example monthly rainfall
    })

    for name, model in models.items():
        prediction = model.predict(new_data)
        print(f"{name} prediction: {prediction[0]}")

    # XGBoost prediction
    dnew_data = xgb.DMatrix(new_data)
    xgb_prediction = xgb_model.predict(dnew_data)
    print(f"XGBoost prediction: {xgb_prediction[0]}")
