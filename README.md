# StockPricePredictionAI

The aim of this project is to create a deployable model for stock price prediction. In order to make the model deployable, I will use the following architecture:


<img src="Software Architecture.jpg">


Small Changes: 

Instead of the training/model validation node in the training pipeline, I implemented a Hyperparameteroptimization node which contains a training, model validation and result upload to MLflow experiments in his loop. 


Notes: Bei Trainingspipeline kommt noch die Hyperparameteroptimierung mit Optuna, Model Quality Validation mit Backtesting, Model registry mit Gitlab MLFlow CI, ohne Canary Deployment. In der Deployment-Pipeline werden bestehende Modelle mit neu generierten Daten getestet und der Score in MLflow hochgeladen. Und Hyperparameteroptimierung findet nur in der CI statt und für jedes Modell einzeln, Optuna-Schema in der Hyperparameteroptimierung sollte für jedes Modell selbst geschrieben werden.

Frage, Sollte das hochladen in jeder Modellklasse sein? Dann wird tag und alles in der Klasse gesetzt.

Entweder alle Modelle gepullt, nochmal trainiert bei continious training und continous deployment auf validate data reduzieren. Bei Model-pull wird die Hyperparameterkombination bei Retraining gepullt, aber das Modell verändert.


With this pipeline, I want to automize the full model lifecycle.

<img src="MLOps_Lifecycle.png">

- Pre-Commit Hooks hinzufügen mit (Code Style Checks und Unittests)
- Poetry initialisieren


First steps:

1. Initialize poetry

Open the terminal in the project directory and use this commands:

- pip install poetry
- poetry shell
- poetry install --no-root


Für pre-commit-hooks:

- pip install pre-commit


Next steps could be...

- DVC instead of updating the DataFrame to MLflow

- Generalize the data preprocessing node because it only works 

pre-commit:


repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v3.4.0
  hooks:
  - id: trailing-whitespace
  - id: end-of-file-fixer
  - id: check-yaml
  - id: check-added-large-files

- repo: https://github.com/psf/black
  rev: 21.12b0
  hooks:
  - id: black
    args: ['--line-length', '88']

- repo: https://github.com/pre-commit/mirrors-pylint
  rev: v2.6.0
  hooks:
  - id: pylint
    args: ['--disable=C', '--disable=W', '--disable=R']

- repo: https://github.com/pre-commit/mirrors-isort
  rev: v5.10.0
  hooks:
  - id: isort
    args: ['--profile', 'black']

- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.1.9
  hooks:
  - id: ruff
    args: ['--fail-on-error']

