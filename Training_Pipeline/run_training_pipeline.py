from utils.pipeline import Pipeline
from training_pipeline.data_preprocessing import DataPreprocessing
from training_pipeline.hyperparameter_optimization import HyperParameterOptimization

data_preprocessing = DataPreprocessing()
hyperparameter_optimization = HyperParameterOptimization()

steps = [data_preprocessing, hyperparameter_optimization]
pipeline = Pipeline(steps=steps)

"""
    Ziel irgendwie dieses Pipeline-Design zu schaffen, dass der Input von einem, der Output von den anderen ist.
"""

