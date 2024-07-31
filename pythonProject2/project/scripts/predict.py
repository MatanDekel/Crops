from flask import Flask, request, jsonify
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from project.utils.model_utils import evaluate_model, evaluate_xgboost
from project.utils.data_utils import load_data, preprocess_data

app = Flask(__name__)


def pred(month, year, crop, season, region):
    df = load_data(fr"D:\Final project\pythonProject2\project\data\evaluation_results_{region}_{season}_{crop}.xlsx")
    features = df[['month', 'year', 'temp', 'humi', 'monthly_rainfall_mm', 'region_encoded', 'season_encoded']]
    target = df['amount']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': joblib.load("models/linear_regression.joblib"),
        'Decision Tree': joblib.load("models/decision_tree.joblib"),
        'Random Forest': joblib.load("models/random_forest.joblib"),
    }
    xgb_model = xgb.Booster()
    xgb_model.load_model("models/xgboost_model.json")

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

    xgb_mae, xgb_mse, xgb_rmse, xgb_r2 = evaluate_xgboost(xgb_model, X_test, y_test)
    evaluation_results.append({
        'Model': 'XGBoost',
        'MAE': xgb_mae,
        'MSE': xgb_mse,
        'RMSE': xgb_rmse,
        'R^2': xgb_r2
    })

    evaluation_df = pd.DataFrame(evaluation_results)
    print("Evaluation Results:")
    print(evaluation_df)

    evaluation_df.to_excel(
        fr"D:\Final project\pythonProject2\project\data\evaluation_results_{region}_{season}_{crop}.xlsx", index=False)
    print(fr"D:\Final project\pythonProject2\project\data\evaluation_results_{region}_{season}_{crop}.xlsx")

    new_data = pd.DataFrame({
        'month': [month],
        'year': [year],
        'desc': [crop],
    })

    predictions = {}
    for name, model in models.items():
        prediction = model.predict(new_data)
        predictions[name] = prediction[0]

    dnew_data = xgb.DMatrix(new_data)
    xgb_prediction = xgb_model.predict(dnew_data)
    predictions['XGBoost'] = xgb_prediction[0]

    return predictions, evaluation_results


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    month = data['month']
    year = data['year']
    crop = data['crop']
    season = data['season']
    region = data['region']

    predictions, evaluation_results = pred(month, year, crop, season, region)

    response = {
        'predictions': predictions,
        'evaluation_results': evaluation_results
    }
    return jsonify(response)




