from pathlib import Path

ROOT = Path(".").parent
DATA = ROOT / "data"
EXTRACTED = DATA / "extracted"

RAW_DATA = DATA / "raw"

SORTED_DATA = DATA / "sorted"
SORTED_HEALTHY = SORTED_DATA / "healthy"
SORTED_STROKE = SORTED_DATA / "stroke"

RAW_PLOTS = DATA / "raw_plots"
SORTED_PLOTS = DATA / "sorted_plots"

# EXTRACTED
OPTIMAL_PARAMS_FILE = EXTRACTED / "optimal_params.csv"
RQA_METRICS_FILE = EXTRACTED / "rqa_metrics.csv"
ML_FEATURES_FILE = EXTRACTED / "ml_features.csv"
FEATURES_RAW_FILE = EXTRACTED / "features_raw.csv"