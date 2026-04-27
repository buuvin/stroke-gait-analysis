"""Build ML-ready feature table by merging handcrafted and RQA metrics.

Notes
-----
This script performs three stages:
1) Extract time/frequency features from raw COP files.
2) Load precomputed RQA metrics.
3) Merge on shared metadata keys and write ``ml_features.csv``.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os

from features.extract_time_features import extract_time_domain_features
from features.extract_freq_features import extract_frequency_domain_features
from features.features_utils import parse_metadata_from_filename, patient_side

from paths import RAW_DATA, ML_FEATURES_FILE, RQA_METRICS_FILE

rows = []
# Stage 1: extract handcrafted features from each raw COP file.
for file in os.scandir(RAW_DATA):
    print("PROCESSING FILE: ", file.name)
    with open(file, 'r') as data_file:
        data = np.loadtxt(data_file, dtype = np.float64)
    
    time = extract_time_domain_features(data)
    meta = parse_metadata_from_filename(file.name, patient_side)
    freq = extract_frequency_domain_features(data)

    row = {**meta, **time, **freq}
    rows.append(row) 
    print("FINISHED FILE: ", file.name)


df_features = pd.DataFrame.from_records(rows)

# Stage 2: load RQA feature table produced by the RQA pipeline.
rqa_features = pd.read_csv(RQA_METRICS_FILE)
rqa_features = rqa_features.rename(columns={"file": "filename"})

rqa_features["affected_side"] = rqa_features["affected_side"].fillna("")

# Stage 3: merge both feature families into one ML table.
features_raw = df_features.merge(rqa_features, on = ['filename', 'category', 'eye_condition', 'affected_side', 'cop_type', 'axis'], how = 'inner')

features_raw.to_csv(ML_FEATURES_FILE, index=False)