import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    '''
    Configuration for saving the trained model file.
    '''
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        '''
        Initializes the model trainer configuration.
        '''
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        '''
        Function to train and evaluate multiple models with hyperparameter tuning.
        - Splits train and test data.
        - Trains models with hyperparameters using cross-validation.
        - Evaluates models and selects the best one.
        '''
        try:
            logging.info("Splitting train and test data into features and target variables.")
            
            # Splitting the train and test data
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Define the models to be trained
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting": CatBoostClassifier(verbose=False)
            }

            # Define the hyperparameter grid
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss']  # Modified for classification
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'algorithm': ['SAMME']
                }
            }

            logging.info("Starting model evaluation with hyperparameter tuning.")
            
            # Evaluate models with cross-validation and return the report
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # Identify the best model by score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with an accuracy score above the threshold of 0.6.")

            logging.info(f"Saving the best model: {best_model_name} to {self.model_trainer_config.trained_model_file_path}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions with the best model
            logging.info("Predicting on the test dataset using the best model.")
            predictions = best_model.predict(X_test)

            # Calculate accuracy score
            accuracy = accuracy_score(y_test, predictions)
            logging.info(f"Accuracy of the best model on the test data: {accuracy}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
