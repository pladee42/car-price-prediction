# Car Price Prediction

This repository contains machine learning models designed to predict car prices using various algorithms. The models are trained and evaluated using a dataset from a Kaggle competition. Several regression models such as `CatBoost`, `XGBoost`, `LightGBM`, and `Random Forest` are used to build and evaluate the prediction accuracy.

## Models

The following models are implemented:
- **CatBoost Regressor**
- **LightGBM Regressor**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **SVR (Support Vector Regressor)**

The models are trained on the car dataset and evaluated using metrics like `Mean Absolute Error (MAE)`, `Mean Squared Error (MSE)`, and `R-squared`.

## Installation

To get started, make sure you have the required dependencies installed.

### Using `pip`

```bash
pip install catboost lightgbm xgboost scikit-learn seaborn matplotlib pandas numpy
```

## Usage

After setting up the environment, you can use the notebook to train models and evaluate their performance.

1. **Load the Dataset**  
   The dataset can be loaded using pandas `read_csv()` or directly from Kaggle.

2. **Train a Model**  
   For example, to train a `CatBoost` model:
   ```python
   model = CatBoostRegressor()
   model.fit(X_train, y_train)
   ```

3. **Evaluate the Model**  
   Use metrics like Mean Absolute Error (MAE) and R-squared to evaluate the model:
   ```python
   y_pred = model.predict(X_test)
   print(mean_absolute_error(y_test, y_pred))
   print(r2_score(y_test, y_pred))
   ```

## Results

The notebook contains various sections to evaluate the performance of different models on the car price prediction task. These include the following:
- Data preprocessing
- Model training
- Hyperparameter tuning
- Model evaluation
