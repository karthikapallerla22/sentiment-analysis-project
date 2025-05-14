import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('stopwords')

# Sample dataset
data = {'text': ["I loved the movie", "Terrible experience", "It was okay", "Wonderful film", "Wouldn't recommend", 
                 "Average acting", "Worst movie", "A masterpiece", "Mediocre", "Brilliant acting"], 
        'label': [1, 0, 2, 1, 0, 2, 0, 1, 2, 1]}  # Labels: 0=Negative, 1=Positive, 2=Neutral
df = pd.DataFrame(data)

# Text cleaning function: Remove punctuation, stopwords, and apply stemming
def clean_text(t):
    return ' '.join([PorterStemmer().stem(w) for w in t.lower().translate(str.maketrans('', '', string.punctuation)).split() if w not in stopwords.words('english')])

df['cleaned'] = df['text'].apply(clean_text)

# Feature extraction using TF-IDF
X = TfidfVectorizer().fit_transform(df['cleaned'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with class weights to handle class imbalance
model = LogisticRegression(max_iter=200, class_weight='balanced').fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Classification report with zero_division to avoid warnings
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive", "Neutral"], zero_division=1))

# Confusion matrix visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=["Neg", "Pos", "Neu"], yticklabels=["Neg", "Pos", "Neu"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
