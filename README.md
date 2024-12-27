The project involves analyzing a time series of Bitcoin prices from January 1, 2015, to January 14, 2024, followed by preparing a predictive model that forecasts the Bitcoin price 3 days ahead.

The “notebooks” folder contains raw data analysis, feature creation, and data preparation for training. The work was conducted in Jupyter notebooks.

The Data_cleaning.ipynb file includes a preliminary data analysis, transformation, and preparation for feature engineering.
The Feature_Engineering.ipynb file contains feature engineering steps. Since the data represents a time series, it was analyzed in terms of seasonality, stationarity, lagged values, and aggregation. Based on these analyses, additional features were created to improve the model’s performance.
The Model_training_data_preparation.ipynb file focuses on preparing the data for training.

The “data” folder contains both the original and the processed data. Due to the size of the dataset, no cloud storage services such as AWS S3 were used.

The “scripts” folder contains scripts with code for hyperparameter tuning and model training.
The XGboost_hyperparameter_tuning_final.py file handles hyperparameter tuning using RandomizedSearchCV. MLFlow was used to monitor the results. The outcome of the exercise was saved as a CSV file for further utilization.

![1](https://github.com/user-attachments/assets/80b1f2db-709b-4f42-8589-c7fb6ae0c1fd)

The results of the exercise have been saved as a CSV file, which enables further use.

![2](https://github.com/user-attachments/assets/c8e8cf6b-d22f-485f-8bd1-9173c71f3cd1)

The XGboost_training_model_final.py file uses the saved hyperparameters and trains the model.

![3](https://github.com/user-attachments/assets/897c4dd3-519b-441b-93fb-75d08efb9361)

The trained model has been saved along with other essential files required for production deployment (e.g., a .yaml file specifying the libraries used).

![4](https://github.com/user-attachments/assets/24ff85e2-22ec-4731-a1d2-90f2054979b9)
