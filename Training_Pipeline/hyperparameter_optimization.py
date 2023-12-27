import optuna
import pandas as pd

from models.varma_model import VARMAModel
from utils.evaluation import backtest
from utils.pipeline_node import PipelineNode

class HyperParameterOptimization(PipelineNode):
    """
    Hyperparameter optimization node for the training pipeline.
    """

    def __init__(self, model_scheme: str) -> None:
        """
        Hyperparameter optimization node for the training pipeline.

        Parameters:

        input (pd.DataFrame): preprocessed input for the Hyperparameter optimization.
        model_scheme (str): model to which the hyperparamater should be tuned.
        """
        super().__init__()
        self.model_scheme = model_scheme

    def process(self, input):
        super().process(input=input)
        if self.model_scheme == "varma":
            result = self._hyperparameter_optimization_for_varma(data=input)
        return result
    
    def _hyperparameter_optimization_for_varma(self, data: pd.DataFrame):
        """
        Hyperparameteroptimization with the optuna framework.

        Parameters:

        data (pd.DataFrame): data for the Backtesting.

        Return:

        model with optimal hyperparamaters.
        """

        def objective(trial):
            p = trial.suggest_int("p", 0, 10)
            d = trial.suggest_int("d", 0, 10)
            parameters = {"order": (p, d)}
            model = VARMAModel()
            return backtest(data=data, model=model, hyperparameters=parameters)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2)
        # replace it with the optimal model from the trials and try to reuse it but it is complicated because it is backtested
        best_params = study.best_trial.params
        best_model = VARMAModel()
        best_model.train(train_data=data, parameters=best_params)
        return best_model.model
