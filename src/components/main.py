import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

datasets = {
    'Heart': 'EDA/data/Heart.csv',
    'Diabetes': 'EDA/data/Diabetes.csv',
    'Liver': 'EDA/data/Liver.csv',
    'Stroke': 'EDA/data/Stroke.csv'
}

targets = {
    'Heart': 'target',
    'Diabetes': 'target',
    'Liver': 'target',
    'Stroke': 'target'
}

numerical_columns = {
    'Heart': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
    'Diabetes': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    'Liver': ["Age", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
                "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
                "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio"],
    'Stroke': ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
}

categorical_columns = {
    'Heart': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
    'Diabetes': [],
    'Liver': ["Gender"],
    'Stroke': ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
}

for disease, dataset_path in datasets.items():
    try:
        logging.info(f"Processing dataset for {disease}")

        # Data Ingestion
        data_ingestion = DataIngestion(dataset_path)
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        target_column = targets[disease]
        num_columns = numerical_columns[disease]
        cat_columns = categorical_columns[disease]

        data_transformation = DataTransformation(disease)
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data, test_data, target_column, num_columns, cat_columns
        )

        # Model Training
        model_file_path = os.path.join("artifacts", f"{disease}_model.pkl")
        model_trainer = ModelTrainer(model_file_path)
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Best model for {disease} achieved accuracy: {accuracy}")

    except CustomException as e:
        logging.error(f"CustomException: Error processing dataset for {disease}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing dataset for {disease}: {e}")
