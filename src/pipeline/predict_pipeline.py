import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        '''
        Takes in features and performs prediction using the pre-trained model and preprocessor.
        '''
        try:
            logging.info("Starting the prediction pipeline.")
            
            # Define paths for the model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info(f"Loading model from {model_path}")
            model = load_object(file_path=model_path)

            logging.info(f"Loading preprocessor from {preprocessor_path}")
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Preprocessing the input features.")
            # Transform features using the preprocessor
            data_scaled = preprocessor.transform(features)

            logging.info("Performing predictions.")
            # Make predictions using the loaded model
            preds = model.predict(data_scaled)

            logging.info("Prediction pipeline completed successfully.")
            return preds
        
        except Exception as e:
            logging.error("Error occurred in the prediction pipeline.")
            raise CustomException(e, sys)


class CustomData:
    '''
    A class to handle custom user input and convert it into a dataframe suitable for the model.
    '''
    def __init__(self,
                 age: int,
                 job: str,
                 marital: str,
                 education: str,
                 default: str,
                 balance: int,
                 housing: str,
                 loan: str,
                 contact: str,
                 day: int,
                 month: str,
                 duration: int,
                 campaign: int,
                 pdays: int,
                 previous: int,
                 poutcome: str):
        
        self.age = age
        self.job = job
        self.marital = marital
        self.education = education
        self.default = default
        self.balance = balance
        self.housing = housing
        self.loan = loan
        self.contact = contact
        self.day = day
        self.month = month
        self.duration = duration
        self.campaign = campaign
        self.pdays = pdays
        self.previous = previous
        self.poutcome = poutcome

    def get_data_as_data_frame(self):
        '''
        Convert the input data into a Pandas DataFrame for model predictions.
        '''
        try:
            logging.info("Converting custom input data into a DataFrame.")
            
            # Create a dictionary from input data
            custom_data_input_dict = {
                "age": [self.age],
                "job": [self.job],
                "marital": [self.marital],
                "education": [self.education],
                "default": [self.default],
                "balance": [self.balance],
                "housing": [self.housing],
                "loan": [self.loan],
                "contact": [self.contact],
                "day": [self.day],
                "month": [self.month],
                "duration": [self.duration],
                "campaign": [self.campaign],
                "pdays": [self.pdays],
                "previous": [self.previous],
                "poutcome": [self.poutcome]
            }

            # Convert to DataFrame
            data_frame = pd.DataFrame(custom_data_input_dict)
            logging.info("Data conversion to DataFrame successful.")
            return data_frame

        except Exception as e:
            logging.error("Error occurred while converting input data to DataFrame.")
            raise CustomException(e, sys)
