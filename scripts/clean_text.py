import os
import re
import unicodedata

import pandas as pd


URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
HTML_RE = re.compile(r'<.*?>')
CONTROL_RE = re.compile(r'[\r\n\t]+')
WHITESPACE_RE = re.compile(r'\s+')
MAX_CHARS = 5000


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)

    s = unicodedata.normalize('NFKC', s)
    s = URL_RE.sub(' ', s)
    s = EMAIL_RE.sub(' ', s)
    s = HTML_RE.sub(' ', s)
    s = CONTROL_RE.sub(' ', s)
    s = WHITESPACE_RE.sub(' ', s)
    s = s.strip()
    s = s.lower()
    return s


def maybe_truncate(s: str) -> str:
    if len(s) > MAX_CHARS:
        return s[:MAX_CHARS]
    return s


def clean_dataset(data_path: str, out_path: str) -> None:
    print('Reading', data_path)
    df = pd.read_csv(data_path)
    initial = len(df)

    text_col = 'statement' if 'statement' in df.columns else [c for c in df.columns if df[c].dtype == object][0]

    before_dropna = len(df)
    df = df.dropna(subset=[text_col])
    dropped_na = before_dropna - len(df)

    df[text_col] = df[text_col].astype(str).map(clean_text)
    truncated = int((df[text_col].map(len) > MAX_CHARS).sum())
    df[text_col] = df[text_col].map(maybe_truncate)

    before_empty = len(df)
    df = df[df[text_col].str.strip().astype(bool)]
    removed_empty = before_empty - len(df)

    df['char_len'] = df[text_col].map(len)
    df['word_len'] = df[text_col].map(lambda s: len(s.split()))

    print(f'Initial rows: {initial}')
    print(f'Dropped missing statements: {dropped_na}')
    print(f'Removed empty after cleaning: {removed_empty}')
    print(f'Truncated >{MAX_CHARS} chars: {truncated}')
    print('Final rows:', len(df))

    cols_to_save = [c for c in df.columns if c not in ('char_len', 'word_len')]
    df[cols_to_save].to_csv(out_path, index=False)
    print('Wrote cleaned dataset to', out_path)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, 'data', 'dataset.csv')
    out_path = os.path.join(root, 'data', 'dataset_clean.csv')
    clean_dataset(data_path, out_path)


if __name__ == '__main__':
    main()
