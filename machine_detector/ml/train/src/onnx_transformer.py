import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx import update_registered_converter
from skl2onnx.common.utils import check_input_and_output_numbers
from skl2onnx.algebra.onnx_ops import (
    OnnxSlice,
    OnnxSub,
    OnnxDiv,
    OnnxMul,
    OnnxAbs,
)

from src.configurations import FeatureConfigurations


def convert_dataframe_schema(df, drop=None):
    inputs = []
    for k, v in zip(df.columns, df.dtypes):
        if drop is not None and k in drop:
            continue
        if v == "int64":
            t = Int64TensorType([None, 1])
        elif v == "float64":
            t = FloatTensorType([None, 1])
        else:
            t = StringTensorType([None, 1])
        inputs.append((k, t))
    return inputs


class AbsDiffCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def calculate_abs_diff(self, x, y):
        return abs(x - y)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = X.apply(
            lambda x: self.calculate_abs_diff(
                x["air_temperature"], x["process_temperature"]
            ),
            axis=1,
        )
        return x.values.reshape((-1, 1))


def abs_diff_shape_calculator(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    # Gets the input type, the transformer works on any numerical type.
    input_type = operator.inputs[0].type.__class__
    # The first dimension is usually dynamic (batch dimension).
    input_dim = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = input_type([input_dim, 1])


def abs_diff_converter(scope, operator, container):
    # No need to retrieve the fitted estimator, it is not trained.
    # op = operator.raw_operator
    opv = container.target_opset
    X = operator.inputs[0]

    # 100 * (x-y)/y  --> 100 * (X[0] - X[1]) / X[1]

    zero = np.array([0], dtype=np.int64)
    one = np.array([1], dtype=np.int64)
    two = np.array([2], dtype=np.int64)

    # Slice(data, starts, ends, axes)
    x0 = OnnxSlice(X, zero, one, one, op_version=opv)
    x1 = OnnxSlice(X, one, two, one, op_version=opv)

    z = OnnxAbs(
        OnnxSub(x0, x1, op_version=opv),
        op_version=opv,
        output_names=operator.outputs[0],
    )
    z.add_to(scope, container)


class SpeedConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def calculate_power(self, x, y):

        return x * (2 * np.pi / 60) / (x * (2 * np.pi / 60) * y)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = X.apply(
            lambda x: self.calculate_power(x["rotational_speed"], x["torque"]), axis=1
        )
        return x.values.reshape((-1, 1))


def speed_power_shape_calculator(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    # Gets the input type, the transformer works on any numerical type.
    input_type = operator.inputs[0].type.__class__
    # The first dimension is usually dynamic (batch dimension).
    input_dim = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = input_type([input_dim, 1])


def speed_power_converter(scope, operator, container):
    # Retrieve target opset version
    opv = container.target_opset
    X = operator.inputs[0]

    # Define constants
    zero = np.array([0], dtype=np.int64)
    one = np.array([1], dtype=np.int64)
    two = np.array([2], dtype=np.int64)
    # Constant value for (2 * np.pi / 60)
    const_val = np.array(2 * np.pi / 60, dtype=np.float32)

    # Slice operations to get 'rotational_speed' (x0) and 'torque' (x1)
    x0 = OnnxSlice(X, zero, one, one, op_version=opv)
    x1 = OnnxSlice(X, one, two, one, op_version=opv)

    # Calculate the numerator and denominator
    numerator = OnnxMul(x0, const_val, op_version=opv)
    denominator = OnnxMul(numerator, x1, op_version=opv)

    # Perform the division to get the final result
    result = OnnxDiv(
        numerator, denominator, op_version=opv, output_names=operator.outputs[0]
    )

    result.add_to(scope, container)


class TorqueConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def calculate_power(self, x, y):
        return x / (y * (2 * np.pi / 60) * x)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = X.apply(
            lambda x: self.calculate_power(x["torque"], x["rotational_speed"]), axis=1
        )
        return x.values.reshape((-1, 1))


def torque_power_shape_calculator(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    # Gets the input type, the transformer works on any numerical type.
    input_type = operator.inputs[0].type.__class__
    # The first dimension is usually dynamic (batch dimension).
    input_dim = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = input_type([input_dim, 1])


def torque_power_converter(scope, operator, container):
    # Retrieve target opset version
    opv = container.target_opset
    X = operator.inputs[0]

    # Define constants
    zero = np.array([0], dtype=np.int64)
    one = np.array([1], dtype=np.int64)
    two = np.array([2], dtype=np.int64)
    # Constant value for (2 * np.pi / 60)
    const_val = np.array(2 * np.pi / 60, dtype=np.float32)

    # x / (y * (2 * np.pi / 60) * x)
    x0 = OnnxSlice(X, zero, one, one, op_version=opv)
    x1 = OnnxSlice(X, one, two, one, op_version=opv)

    # Calculate the numerator and denominator
    denominator = OnnxMul(OnnxMul(x1, const_val, op_version=opv), x0, op_version=opv)

    # Perform the division to get the final result
    result = OnnxDiv(x0, denominator, op_version=opv, output_names=operator.outputs[0])

    result.add_to(scope, container)


class MulCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def calculate_multiply(self, x, y):
        return x * y

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = X.apply(
            lambda x: self.calculate_multiply(x["tool_wear"], x["process_temperature"]),
            axis=1,
        )
        return x.values.reshape((-1, 1))


def multiply_shape_calculator(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    # Gets the input type, the transformer works on any numerical type.
    input_type = operator.inputs[0].type.__class__
    # The first dimension is usually dynamic (batch dimension).
    input_dim = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = input_type([input_dim, 1])


def multiply_converter(scope, operator, container):
    # Retrieve target opset version
    opv = container.target_opset
    X = operator.inputs[0]

    # Define constants
    zero = np.array([0], dtype=np.int64)
    one = np.array([1], dtype=np.int64)
    two = np.array([2], dtype=np.int64)

    # x / (y * (2 * np.pi / 60) * x)
    x0 = OnnxSlice(X, zero, one, one, op_version=opv)
    x1 = OnnxSlice(X, one, two, one, op_version=opv)

    # Calculate the numerator and denominator
    z = OnnxMul(x0, x1, op_version=opv, output_names=operator.outputs[0])

    z.add_to(scope, container)


class DivCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def calculate_divide(self, x, y):
        return x / y

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = X.apply(
            lambda x: self.calculate_divide(
                x["process_temperature"], x["air_temperature"]
            ),
            axis=1,
        )
        return x.values.reshape((-1, 1))


def divide_shape_calculator(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    # Gets the input type, the transformer works on any numerical type.
    input_type = operator.inputs[0].type.__class__
    # The first dimension is usually dynamic (batch dimension).
    input_dim = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = input_type([input_dim, 1])


def divide_converter(scope, operator, container):
    # Retrieve target opset version
    opv = container.target_opset
    X = operator.inputs[0]

    # Define constants
    zero = np.array([0], dtype=np.int64)
    one = np.array([1], dtype=np.int64)
    two = np.array([2], dtype=np.int64)

    # x / (y * (2 * np.pi / 60) * x)
    x0 = OnnxSlice(X, zero, one, one, op_version=opv)
    x1 = OnnxSlice(X, one, two, one, op_version=opv)

    # Calculate the numerator and denominator
    z = OnnxDiv(x0, x1, op_version=opv, output_names=operator.outputs[0])

    z.add_to(scope, container)


update_registered_converter(
    AbsDiffCalculator,
    "AliasAbsDiffCalculator",
    abs_diff_shape_calculator,
    abs_diff_converter,
)


update_registered_converter(
    SpeedConverter,
    "AliasSpeedConverter",
    speed_power_shape_calculator,
    speed_power_converter,
)


update_registered_converter(
    TorqueConverter,
    "AliasTorqueConverter",
    torque_power_shape_calculator,
    torque_power_converter,
)

update_registered_converter(
    MulCalculator,
    "AliasMulCalculator",
    multiply_shape_calculator,
    multiply_converter,
)

update_registered_converter(
    DivCalculator,
    "AliasDivCalculator",
    divide_shape_calculator,
    divide_converter,
)


def preprocess_product_id(df):
    df["product_id_num"] = pd.to_numeric(df["product_id"].str.slice(start=1))
    df.drop("product_id", axis=1, inplace=True)


categorical_features = FeatureConfigurations.CAT_FEATURES
numerical_features = FeatureConfigurations.NUM_FEATURES

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[("ordinal", OrdinalEncoder(handle_unknown="error"))]
)

# Define ColumnTransformer to include the new features and apply numeric transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "air_process_diff",
            Pipeline(
                [("calculator", AbsDiffCalculator()), ("scaler", numeric_transformer)]
            ),
            ["air_temperature", "process_temperature"],
        ),
        (
            "speed_power",
            Pipeline(
                [("calculator", SpeedConverter()), ("scaler", numeric_transformer)]
            ),
            ["rotational_speed", "torque"],
        ),
        (
            "torque_power",
            Pipeline(
                [("calculator", TorqueConverter()), ("scaler", numeric_transformer)]
            ),
            ["torque", "rotational_speed"],
        ),
        (
            "tool_process",
            Pipeline(
                [("calculator", MulCalculator()), ("scaler", numeric_transformer)]
            ),
            ["tool_wear", "process_temperature"],
        ),
        (
            "temp_ratio",
            Pipeline(
                [("calculator", DivCalculator()), ("scaler", numeric_transformer)]  #
            ),
            ["process_temperature", "air_temperature"],
        ),
        # Apply numeric transformer directly to existing numerical features
        ("num", numeric_transformer, numerical_features),
        # Apply categorical transformer to categorical features
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)
