import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.pipeline_node import PipelineNode

class DataPreprocessing(PipelineNode):
    """
    A class for preprocessing and gathering the data for model training.
    """

    def __init__(self) -> None:
        super().__init__() 

    def process(self, input):
        """
        Hits the process for data gathering and preprocessing.

        Parameters:
        
        input (None)

        Return:
        
        result: preprocessed data for the next node.
        """
        super().process(input=input)
        raw_data = self._datagathering()
        return self._preprocessing(raw_data=raw_data)


    def _datagathering(self) -> pd.DataFrame:
        """
        Gather the raw data through downloading with the yfinance package.

        Return:

        raw_data (pd.DataFrame): raw data

        """
        raw_data = yf.Ticker("^GSPC")
        raw_data = raw_data.history(period="max", interval="1d")

        return raw_data

    def _preprocessing(
        self,
        raw_data: pd.DataFrame,
        standart_scaling: bool = False,
        min_max_scaling: bool = False,
    ) -> pd.DataFrame:
        """
        Preprocess the data for models.

        Parameter:

        raw_data (pd.DataFrame): raw data for preprocessing
        min_max_scaling (bool): data will be min-max scaled if true

        Return:

        preprocessed data (pd.DataFrame)
        """
        assert (standart_scaling is False) or (min_max_scaling is False)
        preprocess_data = raw_data.loc[
            "2023-01-01":
        ].copy()  # start date for cutting DataFrame

        if standart_scaling is True:
            scaler = StandardScaler()
            preprocess_data = scaler.fit_transform(preprocess_data)

        if min_max_scaling is True:
            scaler = MinMaxScaler()
            preprocess_data = scaler.fit_transform(preprocess_data)

        return preprocess_data[["Open", "High", "Low", "Close"]]
