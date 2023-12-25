import pandas as pd
from models.BaseModel import BaseModel
from sklearn.metrics import mean_squared_error
import numpy as np

def backtest(data: pd.DataFrame, model: BaseModel, start:int = 2500, step:int = 250):
    """
        This function evaluates a model with given data based on Backtesting.

        Parameters:
        
        data (pd.DataFrame): Training data for the model
        model (an object with a predict function and a train function): The model for evaluation
        start (int): Amount of training data for the first evaluation
        step (int): Horizon for prediction quality
    
        Return:

        score (float): Evaluation score after Backtesting
    """

    assert (data.shape[0] > start) and (start > 0) and (step > 0) 

    evaluation_scores = []

    for evaluation_step in range(start, data.shape[0], step):
        train = data.iloc[0:evaluation_step].copy()
        test = data.iloc[evaluation_step: evaluation_step + step]
        model.train(train_data = train)
        predictions = model.predict(test)
        evaluation_scores.append(mean_squared_error(y_true=test, y_pred=predictions))

    return np.mean(evaluation_scores)