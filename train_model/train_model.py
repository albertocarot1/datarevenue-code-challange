import logging
import os
from pathlib import Path
from time import time

import click
import pandas as pd
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train-model')


@click.command()
@click.option('--train-set-path')
@click.option('--model-out-file')
def train_model(train_set_path, model_out_file):
    """
    Train the wine predictor, with parameters discovered in hyper-parameter
    tuning phase. The model is then saved for future use.

    Parameters
    ----------
    train_set_path: str
        Path to the train set csv file
    model_out_file: str
        Path where the model file will be saved
    """

    assert '.csv' in train_set_path, f'Received {train_set_path}! ' \
                                     f'Please provide a .csv file'
    hp = {'colsample_bytree': 0.3,
          'gamma': 0.1,
          'learning_rate': 0.1,
          'max_depth': 12,
          'min_child_weight': 7}
    train_set = pd.read_csv(train_set_path)
    train_y = train_set[['points']]
    train_x = train_set.drop(columns=['points'])

    logger.info(f'XGBoost Regression with parameters: {hp}')
    model = XGBRegressor(random_state=42,
                         colsample_bytree=hp['colsample_bytree'],
                         learning_rate=hp['learning_rate'],
                         max_depth=hp['max_depth'],
                         min_child_weight=hp['min_child_weight'],
                         gamma=hp['gamma'])

    logger.info('Training model...')
    started = time()
    model.fit(train_x, train_y)

    logger.info(f'Model trained in {time() - started} seconds')
    os.makedirs(os.path.dirname(model_out_file), exist_ok=True)
    model.save_model(Path(model_out_file))
    logger.info(f'Models saved to {model_out_file}')


if __name__ == '__main__':
    # train_model(
    #     '/home/alberto/PycharmProjects/code-challenge-2020/data_root/processed/train.csv',
    #     '/home/alberto/PycharmProjects/code-challenge-2020/data_root/models/xgbr.model')
    train_model()
