import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from preprocessing import prepare_data
import numpy as np

def plot_feature_importance(filepath, output_dir='plots'):
    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(filepath)
    
    # Best params from previous run
    best_params = {
        'colsample_bytree': 0.7, 
        'learning_rate': 0.05, 
        'max_depth': 7, 
        'n_estimators': 300, 
        'subsample': 0.7,
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    
    print("Training XGBoost with best parameters...")
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Top 20 features
    top_n = 20
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title('Top 20 Feature Importances (XGBoost)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    print(f"Feature importance plot saved to {output_dir}/feature_importance.png")

if __name__ == "__main__":
    plot_feature_importance('project_dataset.csv')
