from flask import Flask, render_template, request, redirect, url_for, flash
from src.pipeline.predict_pipeline import PredictPipeline, HeartData, DiabetesData, StrokeData, LiverData

application = Flask(__name__)   
application.secret_key = 'VIVEK2003'

@application.route('/')
def home():
    return render_template('home.html')


@application.route('/about')
def about():
    return render_template('about.html')

@application.route('/predict/<disease>', methods=['GET', 'POST'])
def predict(disease):
    if request.method == 'POST':
        try:
            # Extract input data and create custom_data instance
            data = request.form
            
            if disease == 'heart':
                custom_data = HeartData(
                    age=int(data.get('age')),
                    sex=data.get('sex'),
                    cp=data.get('cp'),
                    trestbps=int(data.get('trestbps')),
                    chol=int(data.get('chol')),
                    thalach=int(data.get('thalach')),
                    oldpeak=float(data.get('oldpeak')),
                    exang=data.get('exang'),
                    slope=data.get('slope'),
                    ca=int(data.get('ca')),
                    thal=data.get('thal'),
                    restecg=data.get('restecg'),
                    fbs=int(data.get('fbs'))
                )
            elif disease == 'diabetes':
                custom_data = DiabetesData(
                    Pregnancies=int(data.get('Pregnancies')),
                    Glucose=int(data.get('Glucose')),
                    BloodPressure=int(data.get('BloodPressure')),
                    SkinThickness=int(data.get('SkinThickness')),
                    Insulin=int(data.get('Insulin')),
                    BMI=float(data.get('BMI')),
                    DiabetesPedigreeFunction=float(data.get('DiabetesPedigreeFunction')),
                    Age=int(data.get('Age'))
                )
            elif disease == 'liver':
                custom_data = LiverData(
                    Age=int(data.get('Age')),
                    Gender=data.get('Gender'),
                    Total_Bilirubin=float(data.get('Total_Bilirubin')),
                    Direct_Bilirubin=float(data.get('Direct_Bilirubin')),
                    Alkaline_Phosphotase=int(data.get('Alkaline_Phosphotase')),
                    Alamine_Aminotransferase=int(data.get('Alamine_Aminotransferase')),
                    Aspartate_Aminotransferase=int(data.get('Aspartate_Aminotransferase')),
                    Total_Protiens=float(data.get('Total_Protiens')),
                    Albumin=float(data.get('Albumin')),
                    Albumin_and_Globulin_Ratio=float(data.get('Albumin_and_Globulin_Ratio'))
                )
            elif disease == 'stroke':
                custom_data = StrokeData(
                    age=int(data.get('age')),
                    gender=data.get('gender'),
                    hypertension=int(data.get('hypertension')),
                    heart_disease=int(data.get('heart_disease')),
                    ever_married=data.get('ever_married'),
                    work_type=data.get('work_type'),
                    Residence_type=data.get('Residence_type'),
                    avg_glucose_level=float(data.get('avg_glucose_level')),
                    bmi=float(data.get('bmi')),
                    smoking_status=data.get('smoking_status')
                )
            else:
                flash('Invalid disease type selected.')
                return redirect(url_for('home'))

            features = custom_data.get_data_as_data_frame()
            print("Features DataFrame:", features)
            
            pipeline = PredictPipeline(disease=disease)
            prediction = pipeline.predict(features)
            print("Prediction:", prediction)

            # Return result to the same page with prediction
            return render_template(f'{disease}.html', prediction=prediction[0])

        except Exception as e:
            flash(f'Error during prediction: {str(e)}', 'error')
            return redirect(url_for('predict', disease=disease))

    return render_template(f'{disease}.html', prediction=None)

if __name__ == '__main__':
    application.run()
