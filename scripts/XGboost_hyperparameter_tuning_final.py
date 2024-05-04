import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import os

def main():
    # Paths for data loading
    BASE_PATH = r"C:\Users\micha\OneDrive\Pulpit\Project\Bitcoin_pred\data\processed"
    PATH_TO_TRAINING_DATA_X = f"{BASE_PATH}/training_data_X.csv"
    PATH_TO_VALIDATION_DATA_X = f"{BASE_PATH}/validation_data_X.csv"
    PATH_TO_TRAINING_DATA_y = f"{BASE_PATH}/training_data_y.csv"
    PATH_TO_VALIDATION_DATA_y = f"{BASE_PATH}/validation_data_y.csv"

    # Load data
    X_train = pd.read_csv(PATH_TO_TRAINING_DATA_X, parse_dates=['date'], index_col='date')
    X_valid = pd.read_csv(PATH_TO_VALIDATION_DATA_X, parse_dates=['date'], index_col='date')
    y_train = pd.read_csv(PATH_TO_TRAINING_DATA_y, parse_dates=['date'], index_col='date')
    y_valid = pd.read_csv(PATH_TO_VALIDATION_DATA_y, parse_dates=['date'], index_col='date')

    # MLflow session starts
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name="XGBoost Hyperparameter Tuning")
    mlflow.start_run(run_name='XGboost_hyperparameter_tuning_v1')

    # Hyperparameters setup for randomized search
    param_dist = {
        'booster': ['dart', 'gbtree'],
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(50, 500),
        'subsample': uniform(0.5, 0.4),
        'max_depth': randint(3, 15),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'colsample_bylevel': uniform(0.5, 0.5),
        'lambda': uniform(0, 1),
        'alpha': uniform(0, 1),
        'rate_drop': uniform(0.0, 0.2),
        'one_drop': [True, False],
        'num_parallel_tree': randint(1, 5)
    }

    # Configure XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')

    # Time-series cross-validation setup
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform randomized search
    random_search = RandomizedSearchCV(xgb_regressor, param_distributions=param_dist, n_iter=10,
                                       scoring='neg_mean_squared_error', n_jobs=-1, cv=tscv, random_state=42)
    random_search.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      early_stopping_rounds=10, verbose=False)

    # Log best hyperparameters and save the model
    best_hyperparams = random_search.best_params_
    print("Best_hyperparams:", best_hyperparams)
    mlflow.log_params(best_hyperparams)
    mlflow.xgboost.log_model(random_search.best_estimator_, "model")

    # Save the best hyperparameters to a CSV file
    file_path = os.path.join(BASE_PATH, 'best_hyperparams.csv')
    # Convert the dictionary to a DataFrame
    hyperparams_df = pd.DataFrame([best_hyperparams])
    # Save DataFrame to CSV
    hyperparams_df.to_csv(file_path, index=False)

    # Log the CSV file as an artifact
    mlflow.log_artifact(file_path)
    # End MLflow session
    mlflow.end_run()

if __name__ == '__main__':
    main()