from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "personality_pipeline.pkl"
)

model = joblib.load(MODEL_PATH)
app = FastAPI(
    title='Personality Prediction API',
    description='Predict Introvert or Extrovert',
    version='1.0'
)

# Load trained model
model = joblib.load('../saved_models/personality_pipeline.pkl')

class UserInput(BaseModel):
    Time_spent_Alone: int
    Stage_fear: str
    Social_event_attendance: int
    Going_outside: int
    Drained_after_socializing: str
    Friends_circle_size: int
    Post_frequency: int

@app.get('/')
def home():
    return {
        'message': 'Personality Prediction API Running'
    }

@app.post('/predict')
def predict(data: UserInput):

    binary_map = {
        'Yes': 1,
        'No': 0
    }

    input_data = {
        'Time_spent_Alone': data.Time_spent_Alone,
        'Stage_fear': binary_map[data.Stage_fear],
        'Social_event_attendance': data.Social_event_attendance,
        'Going_outside': data.Going_outside,
        'Drained_after_socializing': binary_map[data.Drained_after_socializing],
        'Friends_circle_size': data.Friends_circle_size,
        'Post_frequency': data.Post_frequency
    }

    df = pd.DataFrame([input_data])

    # Feature engineering
    df['social_energy_score'] = (
        df['Social_event_attendance'] +
        df['Going_outside'] +
        df['Friends_circle_size'] +
        df['Post_frequency']
    ) - df['Time_spent_Alone']

    df['isolation_ratio'] = (
        df['Time_spent_Alone'] /
        (df['Friends_circle_size'] + 1)
    )

    df['outdoor_social_score'] = (
        df['Going_outside'] *
        df['Social_event_attendance']
    )

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].max()

    label = 'Extrovert' if prediction == 1 else 'Introvert'

    return {
        'prediction': label,
        'confidence': round(float(probability), 4)
    }