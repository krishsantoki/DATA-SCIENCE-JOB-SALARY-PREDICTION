import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import prepare_data
import time

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"--- {model_name} ---")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)
    
    return {'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def train_and_evaluate(filepath):
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(filepath)
    
    results = []
    
    # 1. Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append(evaluate_model(lr, X_test, y_test, "Linear Regression"))
    
    # 2. Ridge Regression
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(evaluate_model(ridge, X_test, y_test, "Ridge Regression"))
    
    # 3. Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    results.append(evaluate_model(dt, X_test, y_test, "Decision Tree"))
    
    # 4. Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))
    
    # 5. XGBoost with Hyperparameter Tuning
    print("Tuning and Training XGBoost...")
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    print(f"Best XGBoost Params: {grid_search.best_params_}")
    results.append(evaluate_model(best_xgb, X_test, y_test, "XGBoost (Tuned)"))
    
    # 6. CatBoost Regressor
    print("Training CatBoost...")
    from catboost import CatBoostRegressor
    cat = CatBoostRegressor(verbose=0, random_state=42)
    cat.fit(X_train, y_train)
    results.append(evaluate_model(cat, X_test, y_test, "CatBoost"))

    # 7. Neural Network (MLPRegressor)
    print("Training Neural Network (MLPRegressor)...")
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    results.append(evaluate_model(mlp, X_test, y_test, "Neural Network (MLP)"))

    # 8. Stacking Regressor (Ensemble)
    print("Training Stacking Regressor...")
    from sklearn.ensemble import StackingRegressor
    estimators = [
        ('ridge', ridge),
        ('xgb', best_xgb),
        ('mlp', mlp),
        ('cat', cat)
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    stack.fit(X_train, y_train)
    results.append(evaluate_model(stack, X_test, y_test, "Stacking Ensemble"))
    
    # Summary
    results_df = pd.DataFrame(results)
    print("\nFinal Results Summary:")
    print(results_df)
    results_df.to_csv('model_performance_summary.csv', index=False)

if __name__ == "__main__":
    train_and_evaluate('project_dataset.csv')
