import pandas as pd


class BaseModel:
    """
    This is the parent class of all models.
    """

    def __init__(self) -> None:
        pass

    def predict(self, steps: int):
        pass

    def train(self, data: pd.DataFrame, parameters: dict):
        pass

    def upload_model():
        pass
