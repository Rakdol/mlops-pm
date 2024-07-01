import os
import sys

from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import config

from src.processing.preprocessing import (
    create_feature,
    CustomLabelEncoder,
    DomainTransformer,
    CustomScaler,
    ToFrameTransformer,
    NameTransformer,
)

input_pipeline = Pipeline(
    [
        ("LabelEncoder", CustomLabelEncoder(config.CAT_FEATURES)),
        (
            "DomainTransformer",
            DomainTransformer(
                func=create_feature, variable_to_drop=config.DROP_FEATURES
            ),
        ),
        ("MedianImputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("ToFrameTransFormer", ToFrameTransformer(config.FRAME_FEATURES)),
        (
            "CustomScaler",
            CustomScaler(variables=(config.NUM_FEATURES + config.TRANS_FEATURES)),
        ),
        ("NameTransformer", NameTransformer()),
    ]
)

# if __name__ == "__main__":
#     train_data = pd.read_csv(os.path.join(config.DATAPATH, config.TRAIN_FILE))
#     X, y = train_data.drop(config.TARGET, axis=1), train_data[config.TARGET]

#     input_pipeline = Pipeline(
#         [
#             ("LabelEncoder", CustomLabelEncoder(config.CAT_FEATURES)),
#             (
#                 "DomainTransformer",
#                 DomainTransformer(
#                     func=create_feature, variable_to_drop=config.DROP_FEATURES
#                 ),
#             ),
#             ("MedianImputer", SimpleImputer(strategy="median", add_indicator=True)),
#             ("ToFrameTransFormer", ToFrameTransformer(config.FRAME_FEATURES)),
#             (
#                 "CustomScaler",
#                 CustomScaler(config.NUM_FEATURES + config.TRANS_FEATURES),
#             ),
#             ("NameTransformer", NameTransformer()),
#         ]
#     )

#     x = input_pipeline.fit_transform(X)
#     print(input_pipeline.get_feature_names_out())

#     df = pd.DataFrame(x, columns=input_pipeline.get_feature_names_out(), index=X.index)
#     print(df)

#     # Inverse transform example
#     try:
#         original_data = input_pipeline.inverse_transform(x)
#         print(original_data)
#     except AttributeError as e:
#         print(f"Error: {e}")
