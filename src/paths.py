from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EXTRACTED = DATA / "extracted"
RAW_DATA = DATA / "raw"
RESULTS = DATA / "results"
FIGURES = DATA / "figures"

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

#results 
TTEST_EO_FILE = RESULTS / "ttest_eyes_open.csv"
TTEST_EC_FILE = RESULTS / "ttest_eyes_closed.csv"
TTEST_LIMB_FILE = RESULTS / "ttest_limbs.csv"

#figures
TTEST_EO_PLOT = FIGURES / "ttest_eyes_open_top10.png"
TTEST_EC_PLOT = FIGURES / "ttest_eyes_closed_top10.png"
TTEST_LIMB_PLOT = FIGURES / "limb_anova_tukey_top10.png"

def ensure_data_dirs():
    for d in (DATA, EXTRACTED, RESULTS, FIGURES, RAW_PLOTS, SORTED_PLOTS):
        d.mkdir(parents=True, exist_ok=True)
