import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.util import save_object

@dataclass
class DataTransformationConfig:
    preprocessed_data_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "test_preparation_course",
                "lunch",
                "parental_level_of_education",
                "race_ethnicity",
                "gender",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ]
            )

            logging.info("Numerical and categorical pipelines created")

            preprocess_pipeline = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                ]
            )

            return preprocess_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation initiated")

            # Read the train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read the train and test datasets")

            preprocess_pipeline = self.get_data_transformer_object()
            target_column = "math_score"
            numerical_features = ["writing_score", "reading_score"]

            input_features_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info("Fitting the preprocess pipeline")

            input_features_train_df = preprocess_pipeline.fit_transform(
                input_features_train_df
            )
            input_features_test_df = preprocess_pipeline.transform(
                input_features_test_df
            )
             # ✅ Debugging prints
            print("Input Train Shape:", input_features_train_df.shape)
            print("Target Train Shape:", target_feature_train_df.shape)
            print("Input Test Shape:", input_features_test_df.shape)
            print("Target Test Shape:", target_feature_test_df.shape)

            train_arr = np.c_[
                input_features_train_df, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_features_test_df, np.array(target_feature_test_df)]

            print("Final Train Array Shape:", train_arr.shape)
            print("Final Test Array Shape:", test_arr.shape)
            
            logging.info("Data transformation completed")
            save_object(file_path="artifacts/preprocessor.pkl", obj=preprocess_pipeline)


            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)
