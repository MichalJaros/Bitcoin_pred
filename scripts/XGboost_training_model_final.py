import xgboost as xgb
import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow.sklearn
import os

# get arguments from command
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# function to load hyperparameters
def load_hyperparams(file_path):
    df = pd.read_csv(file_path)
    return df.iloc[0].to_dict()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    BASE_PATH = r"C:\Users\micha\OneDrive\Pulpit\Project\Bitcoin_pred\data\processed"
    PATH_TO_TRAINING_DATA_X = f"{BASE_PATH}/training_data_X.csv"
    PATH_TO_VALIDATION_DATA_X = f"{BASE_PATH}/validation_data_X.csv"
    PATH_TO_TRAINING_DATA_y = f"{BASE_PATH}/training_data_y.csv"
    PATH_TO_VALIDATION_DATA_y = f"{BASE_PATH}/validation_data_y.csv"
    PATH_TO_TESTING_DATA_X = f"{BASE_PATH}/testing_data_X.csv"
    PATH_TO_TESTING_DATA_y = f"{BASE_PATH}/testing_data_y.csv"

    # Load data
    X_train = pd.read_csv(PATH_TO_TRAINING_DATA_X, parse_dates=['date'], index_col='date')
    X_valid = pd.read_csv(PATH_TO_VALIDATION_DATA_X, parse_dates=['date'], index_col='date')
    y_train = pd.read_csv(PATH_TO_TRAINING_DATA_y, parse_dates=['date'], index_col='date')
    y_valid = pd.read_csv(PATH_TO_VALIDATION_DATA_y, parse_dates=['date'], index_col='date')
    X_test = pd.read_csv(PATH_TO_TESTING_DATA_X, parse_dates=['date'], index_col='date')
    y_test = pd.read_csv(PATH_TO_TESTING_DATA_y, parse_dates=['date'], index_col='date')

    # Reset index
    X_train.reset_index(drop=True, inplace=True)
    X_valid.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_valid.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # MLflow session starts
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    exp = mlflow.set_experiment(experiment_name="XGboost experiment autolog")
    mlflow.start_run(run_name='XGboost_experiment_autolog_v1',nested=True)

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    # Path to the CSV file with hyperparameters
    hyperparams_path = f"{BASE_PATH}/best_hyperparams.csv"

    # Loading hyperparameters
    hyperparams = load_hyperparams(hyperparams_path)

    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "model.type": "XGBoost",
    }

    mlflow.set_tags(tags)
    mlflow.xgboost.autolog(log_input_examples=True)

    # Model training
    model = xgb.XGBRegressor(
        alpha = hyperparams['alpha'],
        booster = hyperparams['booster'],
        colsample_bylevel = hyperparams['colsample_bylevel'],
        colsample_bytree = hyperparams['colsample_bytree'],
        gamma = hyperparams['gamma'],
        learning_rate = hyperparams['learning_rate'],
        max_depth = int(hyperparams['max_depth']),
        min_child_weight = int(hyperparams['min_child_weight']),
        num_parallel_tree = int(hyperparams['num_parallel_tree']),
        n_estimators = int(hyperparams['n_estimators']),
        one_drop = hyperparams['one_drop'] == 'True',
        rate_drop = hyperparams['rate_drop'],
        subsample = hyperparams['subsample'],
        random_state = 42
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10, verbose=True)

    predicted_qualities = model.predict(X_test)

    (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

    print(f"XGBoost model (max_depth={hyperparams['max_depth']}, n_estimators={hyperparams['n_estimators']}, learning_rate={hyperparams['learning_rate']}):")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")

    # Defining the file path
    file_path = "XGboost_artifacts.csv"

    # Checking if the file exists
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("Sample data, example header\n")
            f.write("1,2,3\n")

    # Logging the file as an artifact in MLflow
    mlflow.log_artifact(file_path)
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()
    run = mlflow.last_active_run()
    if run:
        print("Active run id is {}".format(run.info.run_id))
        print("Active run name is {}".format(run.info.run_name))
    else:
        print("No active run found.")