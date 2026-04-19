from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

def explain_message(text):
    spam_keywords = ["free", "win", "winner", "cash", "prize", "click", "urgent"]

    found = [word for word in spam_keywords if word in text.lower()]

    if found:
        return f"Contains spam indicators: {', '.join(found)}"
    return "No strong spam indicators found."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][prediction]

    result = "🚨 Spam Detected" if prediction == 1 else "✅ Not Spam"

    explanation = explain_message(message)

    return render_template(
        'index.html',
        prediction_text=result,
        confidence=round(prob * 100, 2),
        original=message,
        explanation=explanation
    )

# API (bonus)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json['message']
    cleaned = clean_text(data)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return {
        "prediction": "spam" if prediction == 1 else "ham"
    }

if __name__ == "__main__":
    app.run(debug=True)