import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    '''
    Configuration class for data ingestion paths.
    '''
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        '''
        Initializes the configuration for data ingestion.
        '''
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        Handles the data ingestion process:
        - Reads the dataset from a CSV file.
        - Splits the data into train and test sets.
        - Saves the raw, train, and test datasets to CSV files.
        '''
        logging.info("Entered the data ingestion method or component.")
        try:
            # Read the dataset
            df = pd.read_csv('notebook/data/bank-full.csv', sep=';')
            logging.info('Dataset loaded into a pandas dataframe.')

            # Create directories for storing the datasets if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            # Save raw data to file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to CSV file.")

            # Split the dataset into training and testing sets
            logging.info("Initiating train-test split.")
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            # Save train and test sets to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed successfully. Train and test sets saved.")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Data ingestion process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Data transformation process
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Model training process
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)
