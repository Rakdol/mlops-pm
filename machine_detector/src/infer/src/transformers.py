import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pathlib import Path

path = str(Path(__file__).resolve().parents[2])
sys.path.append(path)

from configurations import (
    FeatureConfigurations,
)


def create_feature(df: pd.DataFrame) -> None:
    df["air_process_diff"] = abs(df["air_temperature"] - df["process_temperature"])
    df["speed_power"] = (
        df["rotational_speed"]
        * (2 * np.pi / 60)
        / (df["rotational_speed"] * (2 * np.pi / 60) * df["torque"])
    )

    df["torque_power"] = df["torque"] / (
        df["rotational_speed"] * (2 * np.pi / 60) * df["torque"]
    )

    df["tool_process"] = df["tool_wear"] * df["process_temperature"]
    df["temp_ratio"] = df["process_temperature"] / df["air_temperature"]
    df["product_id_num"] = pd.to_numeric(df["product_id"].str.slice(start=1))

    df.drop(columns="product_id", inplace=True)


class DomainTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func, variable_to_drop=None):
        self.func = func
        self.variable_to_drop = variable_to_drop
        self.feature_names = None
        self.X = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        self.X = X_copy
        self.func(X_copy)

        for var in self.variable_to_drop:
            try:
                X_copy = X_copy.drop(columns=[var])
            except KeyError:
                pass

        self.feature_names = X_copy.columns
        return X_copy

    def inverse_transform(self, X):
        return self.X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names if self.feature_names is not None else input_features


class ToFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: list):
        self.variables = variables
        self.feature_names = variables

    def fit(self, X, y=None):
        self.X = X.copy()
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.variables)

    def inverse_transform(self, X):
        return self.X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names if self.feature_names is not None else input_features


class NameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = None
        self.variables = None

    def fit(self, X, y=None):
        self.variables = X.columns
        return self

    def transform(self, X):
        X.columns = [re.sub(r"[^a-zA-Z0-9_]+", "_", col) for col in X.columns]
        self.feature_names = list(X.columns)
        return X

    def inverse_transform(self, X):
        X.columns = self.variables

    def get_feature_names_out(self, input_features=None):
        return self.feature_names if self.feature_names is not None else input_features


class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        variables: list,
    ):
        self.variables = variables
        self.feature_names = variables
        self.X = None
        # self.scaler = None
        # self.name = "scaler"

        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.variables])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.variables] = self.scaler.transform(X_copy[self.variables])
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()
        X_copy[self.variables] = self.scaler.inverse_transform(X_copy[self.variables])
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return self.feature_names if self.feature_names is not None else input_features


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.X = None
        self.feature_names = None

    def fit(self, X, y=None):
        self.label_dict = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index
            self.label_dict[var] = {k: i for i, k in enumerate(t, 0)}

        self.feature_names = list(self.label_dict.keys())
        return self

    def transform(self, X):
        X = X.copy()
        self.X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X

    def inverse_transform(self, X):
        return self.X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names if self.feature_names is not None else input_features


def get_input_pipeline():
    input_pipeline = Pipeline(
        [
            ("LabelEncoder", CustomLabelEncoder(FeatureConfigurations.CAT_FEATURES)),
            (
                "DomainTransformer",
                DomainTransformer(
                    func=create_feature,
                    variable_to_drop=FeatureConfigurations.DROP_FEATURES,
                ),
            ),
            ("MedianImputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "ToFrameTransFormer",
                ToFrameTransformer(FeatureConfigurations.FRAME_FEATURES),
            ),
            (
                "CustomScaler",
                CustomScaler(
                    variables=(
                        FeatureConfigurations.NUM_FEATURES
                        + FeatureConfigurations.TRANS_FEATURES
                    )
                ),
            ),
            ("NameTransformer", NameTransformer()),
        ]
    )
    return input_pipeline
