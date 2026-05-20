"""Entry-point script for training and evaluating the CNN-based RQA classifier."""

import random
import torch
from pathlib import Path

from config import RANDOM_SEED
from cnn.cnn_utils import set_seed
from cnn.training import run_subject_kfold_cv, run_subject_split
from cnn.visualization import plot_roc_curve, plot_conf_matrix, plot_roc_curve_cv

set_seed(RANDOM_SEED)

ROOT = Path.cwd().parent
DATA_DIR = ROOT / "plots"
DATA = DATA_DIR / "rqa_plots_512"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

fold_results = run_subject_kfold_cv(  
    root_dir=DATA,
    k=5,
    epochs=15,
    seed=42,
    K=24,
    batch_size=2,
    drop_x=True,
    drop_combined=True,
    drop_empty=True,
    device=device
)


print("Using device:", device)

model_results = run_subject_split(  
    root_dir=DATA,
    epochs=15,
    seed=42,
    batch_size=2,
    drop_x=True,
    drop_combined=True,
    drop_empty=True,
    device=device
)

model = model_results["model"]
model.load_state_dict(model_results["best_state"])

plot_roc_curve(model_results["test_eval_out"])
plot_conf_matrix(model_results["test_eval_out"])

plot_roc_curve_cv(fold_results)