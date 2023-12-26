import optuna
import pandas as pd

from models.base_model import BaseModel
from models.varma_model import VARMAModel
from utils.evaluation import backtest


class HyperParameterOptimization:
    """
    Hyperparameter optimization node for the training pipeline.
    """

    def __init__(self, data: pd.DataFrame, model_scheme: BaseModel) -> None:
        """
        Hyperparameter optimization node for the training pipeline.

        Parameters:

        range (list): range of the hyperparameter optimization space.
        model_scheme (BaseModel): model to which the hyperparamater should be tuned.
        """
        self.model_scheme = model_scheme
        if type(model_scheme) is VARMAModel:
            self.model = self._hyperparameter_optimization_for_varma(data=data)

    def _hyperparameter_optimization_for_varma(self, data: pd.DataFrame):
        """
        Hyperparameteroptimization with the optuna framework.

        Parameters:

        data (pd.DataFrame): data for the Backtesting.

        Return:

        model with optimal hyperparamaters.
        """

        def objective(trial):
            p = trial.suggest_int("p", 0, 20)
            d = trial.suggest_int("d", 0, 20)
            parameters = {"order": (p, d)}
            model = VARMAModel()
            return backtest(data=data, model=model, hyperparameters=parameters)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        return study.best_trial.params
