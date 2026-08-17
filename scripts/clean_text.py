import os
import re
import unicodedata
import pandas as pd

root = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(root, 'data', 'dataset.csv')
out_path = os.path.join(root, 'data', 'dataset_clean.csv')

print('Reading', data_path)
df = pd.read_csv(data_path)
initial = len(df)

# Ensure text column
text_col = 'statement' if 'statement' in df.columns else [c for c in df.columns if df[c].dtype == object][0]

# Drop missing statements
before_dropna = len(df)
df = df.dropna(subset=[text_col])
dropped_na = before_dropna - len(df)

# Normalization functions
url_re = re.compile(r'https?://\S+|www\.\S+')
email_re = re.compile(r'\S+@\S+')
html_re = re.compile(r'<.*?>')
control_re = re.compile(r'[\r\n\t]+')
whitespace_re = re.compile(r'\s+')

truncated = 0
max_chars = 5000

def clean_text(s):
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize('NFKC', s)
    s = url_re.sub(' ', s)
    s = email_re.sub(' ', s)
    s = html_re.sub(' ', s)
    s = control_re.sub(' ', s)
    s = whitespace_re.sub(' ', s)
    s = s.strip()
    s = s.lower()
    return s

# Apply cleaning
df[text_col] = df[text_col].astype(str).map(clean_text)

# Truncate very long texts
def maybe_truncate(s):
    global truncated
    if len(s) > max_chars:
        truncated += 1
        return s[:max_chars]
    return s

df[text_col] = df[text_col].map(maybe_truncate)

# Remove empty after cleaning
before_empty = len(df)
df = df[df[text_col].str.strip().astype(bool)]
removed_empty = before_empty - len(df)

# Recompute lengths
df['char_len'] = df[text_col].map(len)
df['word_len'] = df[text_col].map(lambda s: len(s.split()))

print(f'Initial rows: {initial}')
print(f'Dropped missing statements: {dropped_na}')
print(f'Removed empty after cleaning: {removed_empty}')
print(f'Truncated >{max_chars} chars: {truncated}')
print('Final rows:', len(df))

# Save cleaned csv
cols_to_save = [c for c in df.columns if c not in ('char_len','word_len')]
df.to_csv(out_path, index=False)
print('Wrote cleaned dataset to', out_path)
