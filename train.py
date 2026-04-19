import pandas as pd
import re
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure model folder exists
os.makedirs("model", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Load dataset
df = pd.read_csv('data/spam.csv')

# Rename columns
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['message'] = df['message'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)

nb_model.fit(X_train_vec, y_train)
lr_model.fit(X_train_vec, y_train)

# Evaluate
nb_acc = accuracy_score(y_test, nb_model.predict(X_test_vec))
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_vec))

print("Naive Bayes Accuracy:", nb_acc)
print("Logistic Regression Accuracy:", lr_acc)

# Save best model (LR)
joblib.dump(lr_model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

# Confusion matrix
cm = confusion_matrix(y_test, lr_model.predict(X_test_vec))

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('static/confusion_matrix.png')

print("✅ Training complete. Model saved.")