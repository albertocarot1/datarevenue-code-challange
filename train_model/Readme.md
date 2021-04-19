# Train model task

This script will train the wine score predictor XGBoost regression model, 
and save it as a file. The paramters for the model are discovered through
grid search hyper-parameter tuning, whose results can be found in the notebook `DataExploration.ipynb`

```
Usage: train_model.py [OPTIONS]

  Train the wine predictor, with parameters discovered in hyper-parameter
  tuning phase. The model is then saved for future use.

Options:
  --train-set-path TEXT  Path to the train set csv file  [required]
  --model-out-file TEXT  Path where the model file will be saved  [required]
  --help                 Show this message and exit.

```
