"""Entry-point script for training and evaluating the CNN-based RQA classifier."""

import random
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms

from config import RANDOM_SEED
from paths import SORTED_PLOTS, CNN_FOLD_RESULTS, CNN_MODEL_RESULTS, CNN_ROC_CURVE, CNN_CM, CNN_ROC_CURVE_CV, CNN_GRADCAM
from cnn.cnn_utils import set_seed, is_empty_plot
from cnn.datasets import RQAPlotDataset, collect_items
from cnn.training import run_subject_kfold_cv, run_subject_split
from cnn.visualization import plot_roc_curve, plot_conf_matrix, plot_roc_curve_cv
from cnn.grad_cam import GradCAM, show_gradcam_overlay_only, disable_inplace_relu

set_seed(RANDOM_SEED)

ROOT = Path.cwd().parent
DATA_DIR = ROOT / "plots"
DATA = DATA_DIR / "rqa_plots_512"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

fold_results = run_subject_kfold_cv(  
    root_dir=SORTED_PLOTS,
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

with open(CNN_FOLD_RESULTS, "w") as f:
    for fold_dict in fold_results:
        f.write(f"Fold {fold_dict['fold']}:\n")
        f.write(f"  Train eval: {fold_dict.get('train_eval_out')}\n")
        f.write(f"  Val eval: {fold_dict.get('val_eval_out')}\n")
        f.write(f"  Best eval: {fold_dict.get('best_eval_out')}\n\n")
    f.close()

print("Using device:", device)

model_results = run_subject_split(  
    root_dir=SORTED_PLOTS,
    epochs=15,
    seed=42,
    batch_size=2,
    drop_x=True,
    drop_combined=True,
    drop_empty=True,
    device=device
)

with open(CNN_MODEL_RESULTS, "w") as f:
    f.write(f"Test eval: {model_results.get('test_eval_out')}\n")
    f.close()

model = model_results["model"]
model.load_state_dict(model_results["best_state"])

plot_roc_curve(model_results["test_eval_out"], outfile=Path(CNN_ROC_CURVE))
plot_conf_matrix(model_results["test_eval_out"], outfile=Path(CNN_CM))
plot_roc_curve_cv(fold_results, outfile=Path(CNN_ROC_CURVE_CV))


#GRAD-CAM
items = collect_items(SORTED_PLOTS)
print("items collected")

items = [it for it in items if it["axis"] in ("y", "resultant")]
items = [it for it in items if it["cop_plate"] in ("left", "right")]
items = [it for it in items if not is_empty_plot(it["path"])]

# subject-level split
subject_to_label = {}
for it in items:
    subject_to_label[it["subject_id"]] = int(it["label"])

subjects = list(subject_to_label.keys())
subj_labels = [subject_to_label[s] for s in subjects]

train_idx, test_idx = train_test_split(
    list(range(len(subjects))),
    test_size=0.2,  # or 0.25 depending on your dataset size
    stratify=subj_labels,
    random_state=42
)

test_subject_id  = [subjects[i] for i in test_idx]

test_subjects = [it for it in items if it["subject_id"] in test_subject_id]

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0], std=[1]),
])

test_ds = RQAPlotDataset(test_subjects,   transform=eval_tf)

disable_inplace_relu(model)

for param in model.encoder.backbone[6].parameters():
    param.requires_grad = True

model.eval()
    
target_layer = model.encoder.backbone[6]  # Try layer3 instead of layer4 for more detailed features
cam_extractor = GradCAM(model, target_layer)


img, y, sid, path = test_ds[138]   # or test_ds[0]
input_tensor = img.unsqueeze(0).to(device)   # (1, 1, H, W)

model.eval()
cam, pred_class, probs = cam_extractor.generate(input_tensor)

print("True label:", y)
print("Predicted class:", pred_class)
print("Probabilities:", probs)
print("Plot path:", path)

show_gradcam_overlay_only(img, cam, true_label=y, pred_class=pred_class, probs=probs, outfile=Path(CNN_GRADCAM))