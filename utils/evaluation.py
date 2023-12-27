import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from models.base_model import BaseModel


def backtest(
    data: pd.DataFrame,
    model: BaseModel,
    hyperparameters: dict,
    start: int = 50,
    step: int = 5,
    calculate_metric: callable = mean_squared_error,
) -> float:
    """
    This function evaluates a model with given data based on Backtesting.

    Parameters:

    data (pd.DataFrame): Training data for the model
    model (an object with a predict function and a train function): The model for evaluation
    hyperparameter (dict): A dictionary containing the Hyperparameter of the model.
    start (int): Amount of training data for the first evaluation
    step (int): Horizon for prediction quality
    calculate_metric (callable): A function for the metric. It should take y_true (true labels) and y_pred (prediction labels as an input)

    Return:

    score (float): Evaluation score after Backtesting
    """

    assert (data.shape[0] > start) and (start > 0) and (step > 0)

    evaluation_scores = []

    for evaluation_step in range(start, data.shape[0], step):
        if data.shape[0] - evaluation_step < step:
            break
        train = data.iloc[0:evaluation_step].copy()
        model.train(train_data=train, parameters=hyperparameters)
        test = data.iloc[evaluation_step : evaluation_step + step]
        predictions = model.predict(steps=step)
        evaluation_scores.append(calculate_metric(y_true=test, y_pred=predictions))
    return np.mean(evaluation_scores)
