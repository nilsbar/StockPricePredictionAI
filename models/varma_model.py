import pandas as pd
import statsmodels.api as sm

from models.base_model import BaseModel


class VARMAModel(BaseModel):
    """
    This class provides a SARIMAModel instance with the required Data preprocessing for this problem.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = sm.tsa.SARIMAX

    def train(self, data: pd.DataFrame, parameters: dict) -> None:
        """
        Changes the paramaters of the model and fits to the given data.

        Parameters:

        data (pd.DataFrame): A Dataframe which is used as the input data for the model fitting.
        parameters (dict): A dict which contains the paramters with this format {'order' : (p,d)}.

        Return:

        model: A fitted VarMa model according to the parameters and the data.
        """
        super().train(data=data, parameters=dict)
        self.model = sm.tsa.VARMAX(
            endog=data,
            order=parameters["order"],
        )
        self.model = self.model.fit(disp=False)

    def predict(self, steps: int = 3):
        """
        Predicts future values with the trained model.

        Parameters:

        steps (int): horizon of prediction

        Return:

        predictions
        """
        super().predict(steps=steps)
        return self.model.forecast(steps=steps)
