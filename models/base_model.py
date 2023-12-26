import Pandas as pd


class BaseModel:
    """
    This is the parent class of all models.
    """

    def __init__(self) -> None:
        pass

    def predict(self):
        pass

    def train(self, data: pd.DataFrame, parameters: dict):
        pass

    def upload_model():
        pass