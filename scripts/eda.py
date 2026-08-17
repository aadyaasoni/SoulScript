import pandas as pd
import os

root = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(root, 'data', 'dataset.csv')
print('Reading', data_path)

df = pd.read_csv(data_path)
print('\n=== Head (5 rows) ===')
print(df.head(5).to_string(index=False))

print('\n=== Info ===')
print(df.dtypes)

print('\n=== Missing values per column ===')
print(df.isnull().sum())

print('\n=== Duplicate rows ===')
dup_count = df.duplicated().sum()
print('Duplicates:', dup_count)

# identify text column
text_cols = [c for c in df.columns if c.lower() in ('text', 'statement', 'entry', 'content')]
if not text_cols:
    text_cols = [c for c in df.columns if df[c].dtype == object]

text_col = text_cols[0]
print('\nUsing text column:', text_col)

# class column
label_cols = [c for c in df.columns if c.lower() in ('label','status','mood','category')]
label_col = label_cols[0] if label_cols else None
print('Using label column:', label_col)

print('\n=== Class distribution ===')
if label_col:
    print(df[label_col].value_counts(dropna=False))
else:
    print('No label column found')

# text length stats
print('\n=== Text length stats (chars and words) ===')
df['char_len'] = df[text_col].astype(str).map(len)
df['word_len'] = df[text_col].astype(str).map(lambda s: len(s.split()))
print(df[['char_len','word_len']].describe())

print('\n=== Shortest and longest examples ===')
print('\n-- Shortest --')
print(df.loc[df['char_len'].idxmin()][[text_col, label_col]])
print('\n-- Longest --')
print(df.loc[df['char_len'].idxmax()][[text_col, label_col]])

print('\n=== Sample entries per class (up to 3 each) ===')
if label_col:
    for lbl, g in df.groupby(label_col):
        print('\n--', lbl, '(', len(g), 'examples )--')
        print(g[text_col].astype(str).head(3).to_string(index=False))

print('\nEDA complete')
