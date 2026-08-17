import os
import sys

try:
    import kagglehub
except Exception as e:
    print('kagglehub not installed:', e)
    sys.exit(2)

dataset = "suchintikasarkar/sentiment-analysis-for-mental-health-statements"
print('Downloading dataset:', dataset)
path = kagglehub.dataset_download(dataset)
print('Dataset downloaded to:', path)
