# Evaluate model task

This script will evaluate the previously trained wine score predictor XGBoost 
regression model on the hold out test set. The outcome will be a report in form of a
compiled jupyter notebook (`EvaluateModel.ipynb`).

```
Usage: train_model.py [OPTIONS]

  Train the wine predictor, with parameters discovered in hyper-parameter
  tuning phase. The model is then saved for future use.

Options:
  --train-set-path TEXT  Path to the train set csv file  [required]
  --model-out-file TEXT  Path where the model file will be saved  [required]
  --help                 Show this message and exit.

```
