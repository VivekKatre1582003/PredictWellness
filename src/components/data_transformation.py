import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    disease_name: str

    def __post_init__(self):
        # Ensure the path is created with the provided disease_name
        self.preprocessor_obj_file_path = os.path.join('artifacts', f"preprocessor_{self.disease_name}.pkl")

    @staticmethod
    def get_supported_diseases():
        return ["Heart", "Diabetes", "Liver", "Stroke"]

class DataTransformation:
    def __init__(self, disease_name):
        if disease_name not in DataTransformationConfig.get_supported_diseases():
            raise ValueError(f"{disease_name} is not a supported disease. Supported diseases are: {', '.join(DataTransformationConfig.get_supported_diseases())}")
        self.data_transformation_config = DataTransformationConfig(disease_name)

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:
            # Define numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Define categorical pipeline if there are categorical columns
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ]) if categorical_columns else None

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Define ColumnTransformer
            transformers = [
                ("num_pipeline", num_pipeline, numerical_columns)
            ]
            if cat_pipeline:
                transformers.append(("cat_pipeline", cat_pipeline, categorical_columns))

            preprocessor = ColumnTransformer(transformers=transformers)

            return preprocessor

        except Exception as e:
            raise CustomException(f"Error in data transformation for {self.data_transformation_config.disease_name}: {str(e)}", sys)

    def initiate_data_transformation(self, train_path, test_path, target_column_name, numerical_columns, categorical_columns):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Obtain preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            # Separate input features and target variable
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Correct target variable for Liver dataset
            if self.data_transformation_config.disease_name == 'Liver':
                target_feature_train_df = target_feature_train_df.map({1: 0, 2: 1})
                target_feature_test_df = target_feature_test_df.map({1: 0, 2: 1})

            # Apply preprocessing object
            logging.info("Applying preprocessing object on training and testing dataframes.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features and target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            logging.info(f"Saving preprocessing object.")
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(f"Error in initiating data transformation for {self.data_transformation_config.disease_name}: {str(e)}", sys)
