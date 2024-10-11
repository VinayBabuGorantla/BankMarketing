import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    '''
    Configuration class for file paths in data transformation.
    '''
    preprocessor_ob_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        '''
        Initializes the data transformation configuration.
        '''
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function creates preprocessing pipelines for numeric, categorical, and ordinal features.
        It returns a ColumnTransformer that handles all the preprocessing steps.
        '''
        try:
            # List of numerical, categorical, and ordinal columns
            numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
            categorical_columns = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome']
            ordinal_columns = ['education', 'month']

            # Define the order of categories for ordinal encoding
            education_order = ['unknown', 'primary', 'secondary', 'tertiary']
            month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

            # Pipeline for numeric columns
            numeric_pipeline = Pipeline(steps=[("Scaler", StandardScaler())])

            # Pipeline for categorical columns
            categorical_pipeline = Pipeline(steps=[("One-Hot Encoder", OneHotEncoder(handle_unknown='ignore'))])

            # Pipeline for ordinal columns
            ordinal_pipeline = Pipeline(steps=[("Ordinal Encoder", OrdinalEncoder(categories=[education_order, month_order]))])

            # ColumnTransformer to apply the transformations on the respective columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("Numeric Pipeline", numeric_pipeline, numeric_columns),
                    ("Categorical Pipeline", categorical_pipeline, categorical_columns),
                    ("Ordinal Pipeline", ordinal_pipeline, ordinal_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function initiates the data transformation process:
        - Reads the train and test datasets from CSV files.
        - Applies preprocessing to both datasets.
        - Encodes the target column.
        - Saves the preprocessor object.
        - Returns transformed training and testing arrays, along with the preprocessor object path.
        '''
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test datasets successfully.")

            logging.info("Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            # Target column name
            target_column_name = "y"
            label_encoder = LabelEncoder()

            # Splitting features and target for train and test datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = label_encoder.fit_transform(train_df[target_column_name])

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = label_encoder.transform(test_df[target_column_name])

            logging.info("Applying preprocessing object to training and testing datasets.")

            # Transform input features using the preprocessor
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target labels into a single array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object to a file
            logging.info("Saving the preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path

        except Exception as e:
            raise CustomException(e, sys)
