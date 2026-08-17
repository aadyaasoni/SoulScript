import os
import re
import unicodedata
import joblib
import numpy as np

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    # same cleaning as scripts/clean_text.py
    url_re = re.compile(r'https?://\S+|www\.\S+')
    email_re = re.compile(r'\S+@\S+')
    html_re = re.compile(r'<.*?>')
    control_re = re.compile(r'[\r\n\t]+')
    whitespace_re = re.compile(r'\s+')

    s = unicodedata.normalize('NFKC', s)
    s = url_re.sub(' ', s)
    s = email_re.sub(' ', s)
    s = html_re.sub(' ', s)
    s = control_re.sub(' ', s)
    s = whitespace_re.sub(' ', s)
    s = s.strip()
    s = s.lower()
    return s


def load_artifacts(base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    vec = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    clf = joblib.load(os.path.join(models_dir, 'logreg_tfidf.joblib'))
    le = joblib.load(os.path.join(models_dir, 'label_encoder.joblib'))
    return vec, clf, le


def predict_text(text: str, vec, clf, le):
    cleaned = clean_text(text)
    X = vec.transform([cleaned])
    prob = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(prob))
    pred_label = le.inverse_transform([pred_idx])[0]
    # return label and dict of class->prob
    classes = list(le.classes_)
    probs = {cls: float(prob[i]) for i, cls in enumerate(classes)}
    return pred_label, probs


if __name__ == '__main__':
    print('Loading TF-IDF artifacts and label encoder')
    vec, clf, le = load_artifacts()
    samples = [
        'I had a calm day, enjoyed reading and felt relaxed.',
        'I am so stressed about work and cannot sleep properly lately.',
        'I feel hopeless and empty, nothing seems worth it.'
    ]

    for s in samples:
        label, probs = predict_text(s, vec, clf, le)
        print('\nInput:', s)
        print('Predicted:', label)
        print('Probabilities:')
        for cls, p in probs.items():
            print(f'  {cls}: {p:.4f}')
