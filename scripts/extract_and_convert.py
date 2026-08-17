import zipfile
import os
import sys
import pandas as pd
import json

proj_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(proj_root, 'data')
zip_path = os.path.join(data_dir, 'archive.zip')
extract_dir = os.path.join(data_dir, 'extracted')
output_csv = os.path.join(data_dir, 'dataset.csv')

if not os.path.exists(zip_path):
    print('Zip file not found at', zip_path)
    sys.exit(1)

os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)

print('Extracted to', extract_dir)

# find candidate data files
candidates = []
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        lf = f.lower()
        if lf.endswith('.csv') or lf.endswith('.json') or lf.endswith('.xlsx') or lf.endswith('.xls') or lf.endswith('.txt'):
            candidates.append(os.path.join(root, f))

if not candidates:
    print('No data files found inside the zip.')
    sys.exit(1)

print('Found data files:')
for c in candidates:
    print(' -', c)

dfs = []
for path in candidates:
    try:
        if path.lower().endswith('.csv'):
            df = pd.read_csv(path)
            dfs.append(df)
        elif path.lower().endswith('.json'):
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            # if it's a list of records
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                # try to find top-level list
                first_list = None
                for v in data.values():
                    if isinstance(v, list):
                        first_list = v
                        break
                if first_list is not None:
                    df = pd.json_normalize(first_list)
                else:
                    df = pd.json_normalize([data])
            else:
                df = pd.DataFrame([data])
            dfs.append(df)
        elif path.lower().endswith('.xlsx') or path.lower().endswith('.xls'):
            df = pd.read_excel(path)
            dfs.append(df)
        elif path.lower().endswith('.txt'):
            # simple: load lines as single-column
            with open(path, 'r', encoding='utf-8') as fh:
                lines = [l.strip() for l in fh if l.strip()]
            df = pd.DataFrame({'text': lines})
            dfs.append(df)
    except Exception as e:
        print('Failed to read', path, e)

if not dfs:
    print('No readable dataframes produced from candidates.')
    sys.exit(1)

# If multiple dfs, try to concat aligning columns (outer)
try:
    combined = pd.concat(dfs, ignore_index=True, sort=False)
except Exception as e:
    print('Concat failed:', e)
    # fallback: take first
    combined = dfs[0]

combined.to_csv(output_csv, index=False)
print('Wrote combined CSV to', output_csv)
print('Shape:', combined.shape)
