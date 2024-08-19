import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str

class ModelTrainer:
    def __init__(self, model_file_path):
        self.model_trainer_config = ModelTrainerConfig(model_file_path)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "SVM": SVC(),
                "K-Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False)
            }
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30]
                },
                "Logistic Regression": {
                    'C': [0.1, 1, 10]
                },
                "SVM": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                "K-Neighbors": {
                    'n_neighbors': [3, 5, 7]
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)
