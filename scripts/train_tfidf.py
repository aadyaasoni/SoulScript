import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

root = os.path.dirname(os.path.dirname(__file__))
clean_path = os.path.join(root, 'data', 'dataset_clean.csv')
models_dir = os.path.join(root, 'models')
os.makedirs(models_dir, exist_ok=True)

print('Loading cleaned data from', clean_path)
df = pd.read_csv(clean_path)

text_col = 'statement'
label_col = 'status'

df = df.dropna(subset=[text_col, label_col])
X = df[text_col].astype(str).values
y = df[label_col].astype(str).values

print('Encoding labels')
le = LabelEncoder()
y_enc = le.fit_transform(y)

print('Train/test split')
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

print('Vectorizing with TF-IDF')
vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf = vec.transform(X_test)

print('Training Logistic Regression')
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train_tfidf, y_train)

print('Predicting on test set')
y_pred = clf.predict(X_test_tfidf)

macro_f1 = f1_score(y_test, y_pred, average='macro')
print('\nMacro F1:', macro_f1)

print('\nClassification report:')
report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
print(report)

# Save artifacts
vec_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
clf_path = os.path.join(models_dir, 'logreg_tfidf.joblib')
le_path = os.path.join(models_dir, 'label_encoder.joblib')
metrics_path = os.path.join(models_dir, 'tfidf_metrics.json')

print('Saving vectorizer ->', vec_path)
joblib.dump(vec, vec_path)
print('Saving classifier ->', clf_path)
joblib.dump(clf, clf_path)
print('Saving label encoder ->', le_path)
joblib.dump(le, le_path)

metrics = {'macro_f1': float(macro_f1)}
with open(metrics_path, 'w', encoding='utf-8') as fh:
    json.dump(metrics, fh, indent=2)
print('Saved metrics ->', metrics_path)

print('Done')
