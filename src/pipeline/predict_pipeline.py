import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self, disease):
        self.disease = disease

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", f"{self.disease}_model.pkl")
            preprocessor_path = os.path.join('artifacts', f"preprocessor_{self.disease}.pkl")
            
            # Debugging: Print statements to verify paths and loading
            print(f"Loading model from {model_path}")
            model = load_object(file_path=model_path)
            print(f"Loading preprocessor from {preprocessor_path}")
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Ensure the features are being processed
            print("Transforming features")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            # Debugging: Print prediction result
            print(f"Predictions: {preds}")
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame([self.data])
        except Exception as e:
            raise CustomException(e, sys)

# Disease-specific data classes
class HeartData(CustomData):
    def __init__(self, age, sex, cp, trestbps, chol, thalach, oldpeak, exang, slope, ca, thal, restecg, fbs):
        super().__init__(
            age=age,
            sex=sex,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            thalach=thalach,
            oldpeak=oldpeak,
            exang=exang,
            slope=slope,
            ca=ca,
            thal=thal,
            restecg=restecg,
            fbs=fbs
        )

class DiabetesData(CustomData):
    def __init__(self, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
        super().__init__(
            Pregnancies=Pregnancies,
            Glucose=Glucose,
            BloodPressure=BloodPressure,
            SkinThickness=SkinThickness,
            Insulin=Insulin,
            BMI=BMI,
            DiabetesPedigreeFunction=DiabetesPedigreeFunction,
            Age=Age
        )

class StrokeData(CustomData):
    def __init__(self, age, gender, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
        super().__init__(
            age=age,
            gender=gender,
            hypertension=hypertension,
            heart_disease=heart_disease,
            ever_married=ever_married,
            work_type=work_type,
            Residence_type=Residence_type,
            avg_glucose_level=avg_glucose_level,
            bmi=bmi,
            smoking_status=smoking_status
        )

class LiverData(CustomData):
    def __init__(self, Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio):
        super().__init__(
            Age=Age,
            Gender=Gender,
            Total_Bilirubin=Total_Bilirubin,
            Direct_Bilirubin=Direct_Bilirubin,
            Alkaline_Phosphotase=Alkaline_Phosphotase,
            Alamine_Aminotransferase=Alamine_Aminotransferase,
            Aspartate_Aminotransferase=Aspartate_Aminotransferase,
            Total_Protiens=Total_Protiens,
            Albumin=Albumin,
            Albumin_and_Globulin_Ratio=Albumin_and_Globulin_Ratio
        )
