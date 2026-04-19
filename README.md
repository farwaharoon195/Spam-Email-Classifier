# 📧 AI Spam Detection System

A Machine Learning web application that detects spam messages using NLP and Scikit-learn.

---

## 🚀 Features
- Spam detection using NLP (TF-IDF)
- Logistic Regression model
- Confidence score visualization
- Explainable AI (why prediction is spam)
- REST API endpoint
- Modern UI dashboard

---

## 🧠 Tech Stack
- Python
- Scikit-learn
- Flask
- HTML + CSS

---

## 📊 Model Performance
- Accuracy: ~94% (Logistic Regression)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train.py
python app.py

Open:
http://127.0.0.1:5000/

🌐 API Usage

POST /api/predict

{
  "message": "You won a free prize!"
}

Response:

{
  "prediction": "spam"
}
📌 Project Purpose

Built for internship portfolio and ML learning demonstration.

Test on such messages :

ham,Hey how are you doing?
ham,Are we meeting today?
ham,Call me when you reach home
ham,Don't forget the assignment
ham,Let's go out for dinner
ham,Can you send me the file?
ham,I will call you later
ham,See you tomorrow
ham,What are you doing now?
ham,Meeting is at 5 pm
spam,Win a free iPhone now!!!
spam,Congratulations you won $1000
spam,Claim your reward now click here
spam,Limited time offer buy now
spam,You are selected for lottery
spam,Free entry in contest reply now

👨‍💻 Author

Farwa Haroon