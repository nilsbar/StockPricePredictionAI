# StockPricePredictionAI

The aim of this project is to create a deployable model for stock price prediction. In order to make the model deployable, I will use the following architecture:


<img src="Software Architecture.jpg">


Notes: Bei Trainingspipeline kommt noch die Hyperparameteroptimierung mit Optuna, Model Quality Validation mit Backtesting, Model registry mit Gitlab MLFlow CI, ohne Canary Deployment


With this pipeline, I want to provide the full model lifecycle.

<img src="MLOps_Lifecylce.jpg">

- Pre-Commit Hooks hinzufügen mit (Code Style Checks und Unittests)
- Poetry initialisieren


First steps:

1. Initialize poetry

Open the terminal in the project directory and use this commands:

- pip install poetry 
- poetry shell