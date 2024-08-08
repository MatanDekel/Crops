import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.joblib")
        print(f"{name} model saved.")


def train_xgboost(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.05
    }
    num_boost_round = 500
    xgb_model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    xgb_model.save_model("models/xgboost_model.json")
    print("XGBoost model saved.")
    return xgb_model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return mae, mse, rmse, r2


def evaluate_xgboost(xgb_model, X_test, y_test):
    dtest = xgb.DMatrix(X_test)
    predictions = xgb_model.predict(dtest)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return mae, mse, rmse, r2


def best_model(file_path):
    # Load the data from the first sheet
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # Identify the best models based on each metric
    best_mae = df['MAE'].idxmin()
    best_mse = df['MSE'].idxmin()
    best_rmse = df['RMSE'].idxmin()
    best_r2 = df['R^2'].idxmax()

    # Count the number of times each model is the best for a metric
    model_counts = {
        df.loc[best_mae, 'Model']: 0,
        df.loc[best_mse, 'Model']: 0,
        df.loc[best_rmse, 'Model']: 0,
        df.loc[best_r2, 'Model']: 0
    }

    model_counts[df.loc[best_mae, 'Model']] += 1
    model_counts[df.loc[best_mse, 'Model']] += 1
    model_counts[df.loc[best_rmse, 'Model']] += 1
    model_counts[df.loc[best_r2, 'Model']] += 1

    # Find the model with two or more best metrics
    best_models = [model for model, count in model_counts.items() if count >= 2]

    return best_models


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'


def save_model(model, model_name, region, season, crop):
    filename = f"models/model_{region}_{season}_{crop}.joblib"
    joblib.dump(model, filename)
    print(f"Best model ({model_name}) saved to {filename}")


def train_models(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    models = {
        'linear_regression': LinearRegression(),
        'decision_tree': DecisionTreeRegressor(),
        'random_forest': RandomForestRegressor()
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)

    return models
