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
        'learning_rate': 0.05,
        'n_estimators': 500
    }
    xgb_model = xgb.train(params, dtrain, num_boost_round=500)
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
