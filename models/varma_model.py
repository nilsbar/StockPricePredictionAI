import pandas as pd
import statsmodels.api as sm

from models.base_model import BaseModel


class VARMAModel(BaseModel):
    """
    This class provides a VarMA Model instance with the required Data preprocessing for this problem.
    """

    def __init__(self) -> None:
        """
        This class provides a VarMA Model instance with the required Data preprocessing for this problem.
        """
        super().__init__()
        self.model = sm.tsa.SARIMAX
        mlflow_tag = "VarMa"

    def train(self, train_data: pd.DataFrame, parameters: dict) -> None:
        """
        Changes the paramaters of the model and fits to the given data.

        Parameters:

        data (pd.DataFrame): A Dataframe which is used as the input data for the model fitting.
        parameters (dict): A dict which contains the paramters with this format {'order' : (p,d)}.

        Return:

        model: A fitted VarMa model according to the parameters and the data.
        """
        super().train(train_data=train_data, parameters=dict)
        self.model = sm.tsa.VARMAX(
            endog=train_data,
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
