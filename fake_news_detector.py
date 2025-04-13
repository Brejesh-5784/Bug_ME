import pandas as pd
import numpy as np
import string
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load and label datasets
df_fake = pd.read_csv("/Users/brejesh/Downloads/fake_news_detection/Fake.csv")
df_true = pd.read_csv("/Users/brejesh/Downloads/fake_news_detection/True.csv")

df_fake['label'] = 0  # FAKE = 0
df_true['label'] = 1  # TRUE = 1

# Combine datasets
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# Merge title and text
df['text'] = df['title'] + " " + df['text']
df = df[['text', 'label']]

# 2. Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# 4. Pipeline with hyperparameter tuning
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(solver='liblinear'))
])

params = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(pipeline, param_grid=params, cv=3, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# 5. Evaluation
y_pred = grid.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Save model and vectorizer
joblib.dump(grid.best_estimator_, "model.pkl")
print("✅ Model saved as model.pkl")
