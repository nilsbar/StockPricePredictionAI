from training_pipeline.data_preprocessing import DataPreprocessing
from training_pipeline.hyperparameter_optimization import HyperParameterOptimization
from utils.pipeline import Pipeline

data_preprocessing = DataPreprocessing()
# adjust the algorithm under variable "model_scheme"
hyperparameter_optimization = HyperParameterOptimization(model_scheme="varma")

steps = [data_preprocessing, hyperparameter_optimization]
pipeline = Pipeline(steps=steps)

pipeline.run()
