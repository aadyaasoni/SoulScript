# SoulScript

SoulScript is a Streamlit journaling reflection app that uses lightweight NLP models to identify emotional patterns in a journal entry and combine them with self-reported sleep, stress, and mood signals.

The app runs locally with persisted TF-IDF and logistic regression artifacts in `models/`, so it can provide reflections without calling an external API.

Built by Aadyaa Soni

## Features

- Private in-session journaling interface
- Mood pattern prediction from journal text
- Combined wellness score using text, sleep, stress, and mood inputs
- Probability chart and optional technical details
- Gentle, non-clinical suggestions
- Session history with optional export
- Crisis support prompt for high-risk outputs

## Setup

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mac / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

## Project Structure

- `app.py` - Streamlit application
- `scripts/` - data preparation, analysis, and model training utilities
- `models/` - persisted model artifacts and evaluation reports
- `data/` - local dataset workspace; large source files are excluded from the repository
- `document/` - project blueprint documentation

## Model Artifacts

The app expects these files in `models/`:

- `tfidf_vectorizer.joblib`
- `logreg_tfidf.joblib`
- `label_encoder.joblib`
- `tfidf_metrics.json`
- `tfidf_report.txt`
- `embeddings_report.txt`

## Disclaimer

SoulScript is a reflection tool, not a medical diagnosis or replacement for professional care. If you are struggling or feel unsafe, contact a trusted person, local emergency services, or a mental health professional.
