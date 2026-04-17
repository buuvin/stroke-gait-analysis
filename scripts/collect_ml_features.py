import numpy as np
import pandas as pd
from pathlib import Path
import os

from features.extract_time_features import extract_time_domain_features
from features.extract_freq_features import extract_frequency_domain_features
from features.features_utils import parse_metadata_from_filename, patient_side

from paths import RAW_DATA, ML_FEATURES_FILE

rows = []
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
df_features.to_csv(ML_FEATURES_FILE, index=False)