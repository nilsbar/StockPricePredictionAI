import prophet
from models.base_model import BaseModel
import pandas as pd

class ProphetModel(BaseModel):
    """
    This class provides a VarMA Model instance with the required Data preprocessing for this problem.
    """

    def __init__(self) -> None:
        """
        This class provides a VarMA Model instance with the required Data preprocessing for this problem.
        """
        super().__init__()
        self.model = prophet
        mlflow_tag = "Prophet"

    def train(self, train_data: pd.DataFrame, parameters: dict, correct_format: False) -> None:
        """
        Changes the paramaters of the model and fits to the given data.

        Parameters:

        data (pd.DataFrame): A Dataframe which is used as the input data for the model fitting.
        parameters (dict): A dict which contains the paramters with this format {'order' : (p,d)}.
        correct_format (Bool): Indicator if format of train_data is correct. 1. reset index, renamed with dict {'Open': 'y', 'Date': 'ds'}

        Return:

        model: A fitted VarMa model according to the parameters and the data.
        """
        super().train(train_data=train_data, parameters=dict)

        #set to correct format for prophet model
        if correct_format is False:
            train_data = train_data.reset_index()
            train_data = train_data.rename(columns={'Open': 'y', 'Date': 'ds'}, inplace=True)
        
        self.model = prophet()
        self.model.add_regressor('High', standartize = False)
        self.model.add_regressor('Low', standartize = False)
        self.model.add_regressor('Close', standartize = False)
        self.model.fit(train_data)

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