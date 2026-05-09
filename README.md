# Introvert vs Extrovert Personality Classifier

## Project Overview

This project predicts whether a person is an Introvert or Extrovert using Machine Learning.

The system includes:

- Data preprocessing
- Feature engineering
- Model training and evaluation
- Hyperparameter tuning
- FastAPI backend
- Cloud deployment on Render
- Public prediction API

---

# Problem Statement

Build a machine learning model that predicts whether a person is an Introvert or Extrovert based on behavioral and social characteristics.

---

# Dataset Features

| Feature | Description |
|---|---|
| Time_spent_Alone | Amount of time spent alone |
| Stage_fear | Whether the person has stage fear |
| Social_event_attendance | Frequency of attending social events |
| Going_outside | Frequency of going outside |
| Drained_after_socializing | Whether socializing causes fatigue |
| Friends_circle_size | Number of close friends |
| Post_frequency | Social media posting frequency |
| Personality | Target label (Introvert / Extrovert) |

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | ML preprocessing and evaluation |
| CatBoost | Main classification model |
| FastAPI | API backend |
| Render | Cloud deployment |
| Joblib | Model serialization |
| Matplotlib / Seaborn | Data visualization |

---

# Machine Learning Workflow

### Data Preprocessing

The following preprocessing steps were applied:

- Missing value handling
- Binary encoding for Yes/No features
- Target label encoding
- Feature scaling using StandardScaler
- Pipeline-based preprocessing

---

### Feature Engineering

Additional engineered features were created to improve prediction performance.

### Social Energy Score

Measures overall social engagement level.

```python
social_energy_score = (
    Social_event_attendance +
    Going_outside +
    Friends_circle_size +
    Post_frequency
) - Time_spent_Alone
```

---

### Isolation Ratio

Measures relative isolation behavior.

```python
isolation_ratio = (
    Time_spent_Alone /
    (Friends_circle_size + 1)
)
```

---

### Outdoor Social Score

Measures outdoor social interaction intensity.

```python
outdoor_social_score = (
    Going_outside *
    Social_event_attendance
)
```

---

# Models Evaluated

The following machine learning models were tested:

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble tree-based model |
| CatBoost | Final optimized model |

---

# Final Model

## Selected Model: CatBoost Classifier

CatBoost was selected because it:

- Performs exceptionally well on tabular datasets
- Handles structured features efficiently
- Provides strong generalization
- Reduces overfitting risk
- Achieved the best validation performance

---

# Hyperparameter Tuning

RandomizedSearchCV was used for model optimization.

The following parameters were tuned:

- Learning rate
- Tree depth
- Number of iterations
- Regularization strength

Cross-validation was used to improve model reliability and reduce overfitting.

---

# Model Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

# API Development

The trained model was deployed using FastAPI.

## API Endpoint

```text
POST /predict
```

---

# Example Request

```json
{
  "Time_spent_Alone": 8,
  "Stage_fear": "Yes",
  "Social_event_attendance": 1,
  "Going_outside": 2,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 2,
  "Post_frequency": 1
}
```

---

# Example Response

```json
{
  "prediction": "Introvert",
  "confidence": 0.94
}
```

---

# Deployment

## Cloud Platform

The application was deployed using Render cloud platform.

## Public API URL

```text
https://introvert-extrovert-ml.onrender.com/docs#/default/predict_predict_post
```

## Swagger Documentation

```text
https://introvert-extrovert-ml.onrender.com/docs
```

---

# Deployment Configuration

## Build Command

```bash
pip install -r requirements.txt
```

## Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

# Project Structure

```text
introvert-extrovert-classifier/
│
├── api/
│   ├── main.py
│   └── requirements.txt
│
├── notebook/
│   └── training.ipynb
│
├── saved_models/
│   └── personality_pipeline.pkl
│
├── data/
│   └── personality_dataset.csv
│
├── README.md
└── .gitignore
```

---

# Running Locally

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start FastAPI Server

```bash
uvicorn main:app --reload
```

# API Testing

The API was tested using:

- FastAPI Swagger UI
- Randomized input testing
- Postman API testing

---

# Example Test Cases

## Introvert Example

```json
{
  "Time_spent_Alone": 9,
  "Stage_fear": "Yes",
  "Social_event_attendance": 0,
  "Going_outside": 1,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 1,
  "Post_frequency": 0
}
```

Expected Prediction:

```text
Introvert
```

---

## Extrovert Example

```json
{
  "Time_spent_Alone": 1,
  "Stage_fear": "No",
  "Social_event_attendance": 9,
  "Going_outside": 8,
  "Drained_after_socializing": "No",
  "Friends_circle_size": 15,
  "Post_frequency": 8
}
```

Expected Prediction:

```text
Extrovert
```

---

# Future Improvements

Possible future enhancements:

- SHAP Explainability
- Docker deployment
- CI/CD pipeline
- Frontend dashboard
- Real-time analytics
- Advanced ensemble models

---

# Conclusion

This project demonstrates a complete end-to-end machine learning workflow including:

- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter optimization
- API development
- Cloud deployment

The system successfully predicts personality traits using behavioral and social interaction data.

---

