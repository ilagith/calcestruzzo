from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from typing import Any, Tuple
from xgboost import XGBRegressor


def get_models_results(models: dict, param_grids: dict, x_train: np.ndarray,
                       x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compare Linear regression baseline with models that require tuning.
    If tuning is necessary, a RandomSearchCV is performed and the best
    tuned model is used to retrieve predictions on the test set and to
    calculate the root mean squared error.
    If no tuning is necessary, a LinearRegression is fitted and evaluated
    on the test set.
    In both cases, results are stored in a dictionary with model name and
    rmse on test set.

    :param models: Dictionary of models to compare
    :param param_grids: Dictionary of models and corresponding parameters space
    :param x_train: Training set
    :param x_test: Test set
    :param y_train: Training label
    :param y_test: Test label
    :return results: Dictionary of model name and corresponding RMSE results
    """
    results = {}

    for model_name, model in models.items():
        params_space = param_grids.get(model_name)

        # models to tune
        if params_space:
            fitted_model, tuning_params = random_search_cv(x_train, y_train, model, params_space)
            y_pred = get_predictions(fitted_model, x_test)

        # baseline
        else:
            fitted_model = model.fit(x_train, y_train)
            y_pred = get_predictions(fitted_model, x_test)

        rmse = compute_error(y_test, y_pred, squared=False)
        results[model_name] = [rmse, fitted_model]

    return results


def define_models() -> dict:
    """
    Get dictionary of desired models. LinearRegression will be
    used as a baseline for this project.
    :return: Dictionary with model name as key and model as values
    """
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_jobs=-1),
        'gbm': GradientBoostingRegressor(),
        'xgboost': XGBRegressor(),
        'lightgbm': LGBMRegressor()
    }

    return models


def define_params_space() -> dict:
    """
    Some models need to be tuned with Random Search.
    Defining their parameters space here.
    :return models_params_space: Dict of parameters space with model name
    """
    models_params_space = {
        'random_forest': {
            'max_depth': list(range(5, 7)),
            'min_samples_leaf': list(range(30, 51)),
            'min_samples_split': list(range(2, 20)),
            'n_estimators': list(range(50, 900, 150)),
        },
        'gbm': {
            'max_depth': list(range(1, 3)),
            'n_estimators': list(range(50, 950, 50))
        },
        'xgboost': {
            'learning_rate': list(np.arange(0.1, 0.3, 0.01)),
            'max_depth': list(range(1, 3)),
            'n_estimators': list(range(50, 450, 50)),
            'lambda': list(np.arange(1, 5.5, 0.25)),
            'alpha': list(np.arange(0, 2.5, 0.25)),
        },
        'lightgbm': {
            'num_leaves': list(range(5, 20, 2)),
            'learning_rate': list(np.arange(0.01, 0.2, 0.01)),
            'n_estimators': list(range(5, 160, 3)),
            'reg_lambda': list(np.arange(1, 3, 0.25)),
        }
    }

    return models_params_space


def compute_error(y_true: np.ndarray, y_pred: np.ndarray, squared: bool) -> float:
    """
    Use sklearn mean squared error function without square root
    to compute root mean squared error given true target values
    and predicted values.
    :param y_true: True target values from test set
    :param y_pred: Model predicted values for target
    :param squared: Bool to indicate computing MSE (True) or RMSE (False)
    :return error: Root Mean Squared Error or Mean Squared Error
    """
    error = mean_squared_error(y_true, y_pred, squared=squared)

    return error


def random_search_cv(x_train: np.ndarray, y_train: np.ndarray, model: Any, parameters_grid: dict) -> Tuple[Any, dict]:
    """
    Perform a 10-fold cross validation random search with 50 iterations
    using the parameters space defined and the desired model.
    Hence, return the best estimator according to the random search
    :param x_train: Array of training features
    :param y_train: Array of target feature
    :param model: Model to random search params for
    :param parameters_grid: Dictionary of parameters space
    :return best_performing_model: Best performing model object
    :return best_performing_params: Dict of best performing params used to train best performing model
    """
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_iter=50,
        random_state=123
    )

    rs.fit(x_train, y_train)
    best_performing_model = rs.best_estimator_
    best_performing_params = rs.best_params_

    return best_performing_model, best_performing_params


def get_predictions(fitted_model: Any, x_test: np.ndarray) -> np.ndarray:
    """
    Retrieve predictions with fitted model
    :param fitted_model: Model trained on training set
    :param x_test: Array of test set
    :return predictions: Array of predictions
    """
    predictions = fitted_model.predict(x_test)

    return predictions


def get_feature_importance(input_features: list, model_name: str, fitted_model) -> None:
    """
    Feature importance retrieved depending on the best performing model.
    Plotted on a barchart
    :param input_features: List of column names
    :param model_name: String of model name
    :param fitted_model: Fitted model
    :return: None, plot of feature importance
    """
    if model_name == 'linear_regression':
        variable_importances = fitted_model.coef_
    else:
        variable_importances = fitted_model.feature_importances_

    plt.figure(figsize=(20, 10))
    plt.barh(input_features, variable_importances)
    plt.show()
