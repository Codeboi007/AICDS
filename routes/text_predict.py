import joblib
import numpy as np

# Load the updated Logistic Regression model and vectorizer
model = joblib.load("./models/logistic_model.pkl")
vectorizer = joblib.load("./models/vectorizer.pkl")

def predict(text):
    if not text.strip():
        return "No input", 0

    # Transform input using the saved vectorizer
    X = vectorizer.transform([text])
    
    # Predict label and probabilities
    predicted = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = np.max(probs)

    label = "AI" if predicted == 1 else "Human"
    score = round(confidence * 100, 2)
    return label, score
