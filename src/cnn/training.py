"""Training loops and experiment orchestration for CNN-based RQA classification."""

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import copy
import random

from config import RANDOM_SEED
from cnn.datasets import RQAPlotDataset, collect_items
from cnn.cnn_utils import is_empty_plot
from cnn.models import PlotLevelCNN
from cnn.evaluation import eval_plot_level, evaluate_at_threshold


def train_one_epoch(model, loader, optimizer, criterion, grad_clip=1.0, device="cpu"):
    """Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model being optimized.
    loader : torch.utils.data.DataLoader
        Training dataloader.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model weights.
    criterion : callable
        Loss function used for supervised training.
    grad_clip : float or None, default 1.0
        Maximum gradient norm for clipping; disabled when ``None``.
    device : str or torch.device, default "cpu"
        Device used to move input batches.

    Returns
    -------
    tuple[float, float]
        Mean training loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    for X, y, _, _ in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += int(y.size(0))

    return total_loss / max(n, 1), correct / max(n, 1)

def find_best_threshold(y_true, y_score):
    """Select the threshold that maximizes balanced accuracy.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_score : array-like
        Continuous class scores or probabilities.

    Returns
    -------
    tuple[float, float]
        Best threshold and the corresponding balanced-accuracy score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    bal_scores = (tpr + (1 - fpr)) / 2.0
    best_idx = np.argmax(bal_scores)
    return thresholds[best_idx], bal_scores[best_idx]


def run_subject_kfold_cv(
    root_dir,
    k=5,
    epochs=25,
    seed=42,
    K=24,
    batch_size=2,
    drop_x=True,
    drop_combined=True,
    drop_empty=True,
    device="cpu"
):
    """Run subject-level stratified k-fold cross-validation.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Directory containing the RQA plot images.
    k : int, default 5
        Number of subject-level folds.
    epochs : int, default 25
        Training epochs per fold.
    seed : int, default 42
        Random seed used for split and augmentation reproducibility.
    K : int, default 24
        Unused legacy argument kept for compatibility.
    batch_size : int, default 2
        Batch size used by the data loaders.
    drop_x : bool, default True
        Whether to discard x-axis plots.
    drop_combined : bool, default True
        Whether to discard combined COP plots.
    drop_empty : bool, default True
        Whether to discard empty plots.
    device : str or torch.device, default "cpu"
        Device used for training and evaluation.

    Returns
    -------
    list[dict]
        Per-fold results containing the best metrics and trained model.
    """

    items = collect_items(root_dir)

    if drop_x:
        items = [it for it in items if it["axis"] in ("y", "resultant")]
    if drop_combined:
        items = [it for it in items if it["cop_plate"] in ("left", "right")]
    if drop_empty:
        print("pre empty filter healthy: ", len([it for it in items if it["label"] == 0]))
        print("pre empty filter stroke: ", len([it for it in items if it["label"] == 1]))
        items = [it for it in items if not is_empty_plot(it["path"])]
        print("post empty filter healthy: ", len([it for it in items if it["label"] == 0]))
        print("post empty filter stroke: ", len([it for it in items if it["label"] == 1]))

    subject_to_label = {}
    for it in items:
        subject_to_label[it["subject_id"]] = int(it["label"])

    subjects = np.array(list(subject_to_label.keys()))
    subj_labels = np.array([subject_to_label[s] for s in subjects])

    print("\nSubjects:", len(subjects), "by class:", Counter(subj_labels))
    print("Images:", len(items))

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(subjects, subj_labels), start=1):
        tr_subjects = set(subjects[tr_idx])
        val_subjects = set(subjects[va_idx])

        train_items = [it for it in items if it["subject_id"] in tr_subjects]
        val_items   = [it for it in items if it["subject_id"] in val_subjects]

        train_ds = RQAPlotDataset(train_items, transform=train_tf)
        val_ds   = RQAPlotDataset(val_items,   transform=val_tf)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        tr_counts = Counter([it["label"] for it in train_items])
        va_counts = Counter([it["label"] for it in val_items])
        print(f"\nFold {fold}/{k}")
        print("  Train plots:", len(train_items), "by class:", tr_counts)
        print("  Val plots:", len(val_items), "by class:", va_counts)
        print("  Train subjects:", len(tr_subjects))
        print("  Val subjects:", len(val_subjects))

        model = PlotLevelCNN(emb_dim=128, num_classes=2, dropout=0.3).to(device)

        n0 = tr_counts.get(0, 0)
        n1 = tr_counts.get(1, 0)
        w0 = (n0 + n1) / (2 * n0) if n0 else 1.0
        w1 = (n0 + n1) / (2 * n1) if n1 else 1.0
        weight = torch.tensor([w0, w1], dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=weight)

        model.freeze_backbone()
        print(sum(p.requires_grad for p in model.parameters()))

        backbone_params = list(model.encoder.backbone[7][-1].parameters())
        head_params = [p for n, p in model.named_parameters() if "encoder.backbone" not in n and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": 5e-5},
                {"params": backbone_params, "lr": 0.0},
            ],
            weight_decay=1e-4
        )

        best_auc = -1.0
        best_bal = -1.0
        best_conf = None
        best_sens = None
        best_spec = None
        best_eval_out = None

        for ep in range(1, epochs + 1):
            if ep == 10:
                model.unfreeze_last_block_only()
                optimizer.param_groups[1]["lr"] = 5e-6
                optimizer.param_groups[0]["lr"] = 5e-5
                print(sum(p.requires_grad for p in model.parameters()))
                
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, grad_clip=1.0)
            val_out = eval_plot_level(model, val_loader)

            val_bal = val_out["bal_acc"]
            val_auc = val_out["auc"]
            val_conf = val_out["conf_mat"]
            tp, tn, fp, fn = val_out["stats"]
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0

            if val_auc is None:
                print(f"  Ep {ep:02d} | train loss/acc {tr_loss:.4f}/{tr_acc:.4f} | val bal {val_bal:.4f} | AUC NA | conf {val_conf} | sensitivity {sens:.3f} | specificity {spec:.3f}")
            else:
                print(f"  Ep {ep:02d} | train loss/acc {tr_loss:.4f}/{tr_acc:.4f} | val bal {val_bal:.4f} | AUC {val_auc:.4f} | conf {val_conf} | sensitivity {sens:.3f} | specificity {spec:.3f}")

            score_auc = val_auc if val_auc is not None else -1.0
            if (score_auc > best_auc) or (score_auc == best_auc and val_bal > best_bal):
                best_auc = score_auc
                best_bal = val_bal
                best_conf = val_conf
                best_sens = sens
                best_spec = spec
                best_eval_out = val_out

            if device.type == "mps":
                torch.mps.empty_cache()

        print(f"Fold {fold:02d} | best auc {best_auc:.4f} | best bal {best_bal:.4f} | sensitivity {best_sens:.4f} | specificity {best_spec}")
        fold_results.append({"fold": fold, "best_auc": best_auc, "best_bal": best_bal, "best_conf": best_conf, "best_sens": best_sens, "best_spec": best_spec, "best_eval_out": best_eval_out, "model": model})

    aucs = np.array([r["best_auc"] for r in fold_results], dtype=float)
    bals = np.array([r["best_bal"] for r in fold_results], dtype=float)
    senss = np.array([r["best_sens"] for r in fold_results], dtype=float)
    specs = np.array([r["best_spec"] for r in fold_results], dtype=float)

    print("\n=== PLot-Level CNN CV Summary (best epoch per fold) ===")
    print("AUC: mean", round(float(aucs.mean()), 4), "std", round(float(aucs.std()), 4), "vals", np.round(aucs, 4))
    print("Bal: mean", round(float(bals.mean()), 4), "std", round(float(bals.std()), 4), "vals", np.round(bals, 4))
    print("Sensitivity: mean", round(float(senss.mean()), 4), "std", round(float(senss.std()), 4), "vals", np.round(senss, 4))
    print("Specificity: mean", round(float(specs.mean()), 4), "std", round(float(specs.std()), 4), "vals", np.round(specs, 4))

    return fold_results

def run_subject_split(
    root_dir,
    epochs=15,
    seed=42,
    batch_size=2,
    drop_x=True,
    drop_combined=True,
    drop_empty=True,
    device="cpu"
):
    """Run a single train/validation/test subject split experiment.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Directory containing the RQA plot images.
    epochs : int, default 15
        Training epochs.
    seed : int, default 42
        Random seed used for split and augmentation reproducibility.
    batch_size : int, default 2
        Batch size used by the data loaders.
    drop_x : bool, default True
        Whether to discard x-axis plots.
    drop_combined : bool, default True
        Whether to discard combined COP plots.
    drop_empty : bool, default True
        Whether to discard empty plots.
    device : str or torch.device, default "cpu"
        Device used for training and evaluation.

    Returns
    -------
    dict
        Trained model, epoch history, and best validation/test outputs.
    """

    items = collect_items(root_dir)
    print("items collected")

    if drop_x:
        items = [it for it in items if it["axis"] in ("y", "resultant")]
    if drop_combined:
        items = [it for it in items if it["cop_plate"] in ("left", "right")]
    if drop_empty:
        print("pre empty filter healthy: ", len([it for it in items if it["label"] == 0]))
        print("pre empty filter stroke: ", len([it for it in items if it["label"] == 1]))
        items = [it for it in items if not is_empty_plot(it["path"])]
        print("post empty filter healthy: ", len([it for it in items if it["label"] == 0]))
        print("post empty filter stroke: ", len([it for it in items if it["label"] == 1]))

    subject_to_label = {}
    for it in items:
        subject_to_label[it["subject_id"]] = int(it["label"])

    subjects = list(subject_to_label.keys())
    subj_labels = [subject_to_label[s] for s in subjects]

    print("\nSubjects:", len(subjects), "by class:", Counter(subj_labels))
    print("Images:", len(items))

    train_idx, test_idx = train_test_split(
        list(range(len(subjects))),
        test_size=0.2,
        stratify=subj_labels,
        random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        stratify=[subj_labels[i] for i in train_idx],
        random_state=seed
    )

    train_subject_id = [subjects[i] for i in train_idx]
    val_subject_id = [subjects[i] for i in val_idx]
    test_subject_id  = [subjects[i] for i in test_idx]

    print("  Train subjects:", len(train_subject_id), "by class:", Counter([subj_labels[i] for i in train_idx]))
    print("  Val subjects:", len(val_subject_id), "by class:", Counter(subj_labels[i] for i in val_idx))
    print("  Test subjects:", len(test_subject_id), "by class:", Counter([subj_labels[i] for i in test_idx]))

    train_subjects = [it for it in items if it["subject_id"] in train_subject_id]
    val_subjects = [it for it in items if it["subject_id"] in val_subject_id]
    test_subjects = [it for it in items if it["subject_id"] in test_subject_id]

    tr_counts = Counter([it["label"] for it in train_subjects])

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = RQAPlotDataset(train_subjects, transform=train_tf)
    val_ds   = RQAPlotDataset(val_subjects,   transform=eval_tf)
    test_ds   = RQAPlotDataset(test_subjects,   transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = PlotLevelCNN(emb_dim=128, num_classes=2, dropout=0.3).to(device)

    n0 = tr_counts.get(0, 0)
    n1 = tr_counts.get(1, 0)
    w0 = (n0 + n1) / (2 * n0) if n0 else 1.0
    w1 = (n0 + n1) / (2 * n1) if n1 else 1.0
    weight = torch.tensor([w0, w1], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)

    model.freeze_backbone()
    print(sum(p.requires_grad for p in model.parameters()))

    backbone_params = list(model.encoder.backbone[7][-1].parameters())
    head_params = [p for n, p in model.named_parameters() if "encoder.backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": 5e-5},
            {"params": backbone_params, "lr": 0.0},
        ],
        weight_decay=1e-4
    )

    results = []
    best_auc = -1.0
    best_bal = -1.0
    best_conf = None
    best_eval_out = None
    best_state = None
    best_t = None
    best_sens = None
    best_spec = None

    for ep in range(1, epochs + 1):
        if ep == 10:
            model.unfreeze_last_stage()
            optimizer.param_groups[1]["lr"] = 5e-6
            optimizer.param_groups[0]["lr"] = 5e-5
            print(sum(p.requires_grad for p in model.parameters()))
            
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, grad_clip=1.0, device=device)
        eval_out = eval_plot_level(model, val_loader, device=device)
        ep_thresh, ep_bal = find_best_threshold(eval_out["y_true"], eval_out["y_score"])
        val_thresh_out = evaluate_at_threshold(eval_out["y_true"], eval_out["y_score"], ep_thresh)

        val_bal = val_thresh_out["bal_acc"]
        val_auc = val_thresh_out["auc"]
        val_conf = val_thresh_out["conf_mat"]
        tp, tn, fp, fn = val_thresh_out["stats"]
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        threshold = val_thresh_out["threshold"]

        print(f"  Ep {ep:02d} | train loss/acc {tr_loss:.4f}/{tr_acc:.4f} | val bal {val_bal:.4f} | AUC {val_auc:.4f} | conf {val_conf} | threshold {threshold:.4f} | sens {sens:.4f} | spec {spec:.4f}") # | TPR {va_tpr:.3f} | TNR {va_tnr:.3f} | best t {best_t:.2f}")

        score_auc = val_auc if val_auc is not None else -1.0
        if (score_auc > best_auc) or (score_auc == best_auc and val_bal > best_bal):
            best_auc = score_auc
            best_bal = val_bal
            best_conf = val_conf
            best_eval_out = eval_out
            best_state = copy.deepcopy(model.state_dict())
            best_t = val_thresh_out["threshold"]
            best_sens = sens
            best_spec = spec

        results.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "test_bal": val_bal,
            "test_auc": val_auc,
            "test_conf": val_conf
        })

        if device.type == "mps":
            torch.mps.empty_cache()

    print("\n=== Single Split Summary ===")
    print("Best AUC:", round(float(best_auc), 4))
    print("Best balanced accuracy:", round(float(best_bal), 4))
    print("Best confusion matrix:", best_conf)

    model.load_state_dict(best_state)
    raw_test_out = eval_plot_level(model, test_loader, device=device)
    test_out = evaluate_at_threshold(
        raw_test_out["y_true"],
        raw_test_out["y_score"],
        best_t
    )
    test_bal = raw_test_out["bal_acc"]
    test_auc = raw_test_out["auc"]
    test_cm = raw_test_out["conf_mat"]
    tp, tn, fp, fn = raw_test_out["stats"]
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0
    print(f"\nTest Results:\ntest bal {test_bal:.4f} | AUC {test_auc:.4f} | conf {test_cm} | sens {sens:.4f} | spec {spec:.4f}")
    

    return {
        "model": model,
        "results": results,
        "best_eval_out": best_eval_out,
        "best_state": best_state,
        "test_eval_out": test_out,
        "raw_test_out": raw_test_out,
    }