import Pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from models.base_model import BaseModel


class SARIMAModel(BaseModel):
    """
    This class provides a SARIMAModel instance with the required Data preprocessing for this problem.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = SARIMAX

    def train(self, data: pd.DataFrame, parameters: dict):
        """
        Changes the paramaters of the model and fits to the given data.

        Parameters:

        data (pd.DataFrame): A Dataframe which is used as the input data for the model fitting.
        parameters (dict): A dict which contains the paramters with this format {'order' : (p,d,q), 'seasonal_order' : (P,D,Q,s)}.

        Return:

        model: A fitted Sarimax model according to the parameters and the data.
        """
        super().train(parameters=dict)
        self.model = SARIMAX(
            data=data,
            endog=None,
            order=parameters["order"],
            seasonal_order=parameters["seasonal_order"],
        )
        self.model.fit(disp=False)

    def predict(self):
        """
        Predicts future values with the trained model.

        Return:

        predictions
        """
        super().predict()
        return self.model.get_prediction()
