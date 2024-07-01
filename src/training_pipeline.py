import os
import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import xgboost as xgb
import optuna
from src.processing.data_handling import load_dataset, save_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import config
import src.pipeline as pipe


def train_model():
    train_data = load_dataset(config.TRAIN_FILE)
    X, y = train_data.drop(config.TARGET, axis=1), train_data[config.TARGET]
    preprocessing = pipe.input_pipeline

    lin_reg = make_pipeline(preprocessing, LogisticRegression(max_iter=2000))
    lin_reg.fit(X, y)
    save_pipeline(lin_reg)


if __name__ == "__main__":
    train_model()
