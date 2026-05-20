from sklearn.metrics import roc_auc_score, confusion_matrix
"""Evaluation helpers for CNN-based RQA classification."""

from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve

@torch.no_grad()
def eval_plot_level(model, loader, device="cpu"):
    """Evaluate a CNN at the plot level on a dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        Fitted classification model.
    loader : torch.utils.data.DataLoader
        DataLoader yielding ``(X, y, subject_id, path)`` batches.
    device : str or torch.device, default "cpu"
        Device used to move the batch tensors.

    Returns
    -------
    dict
        Plot-level balanced accuracy, ROC-AUC, confusion matrix, and scores.
    """
    model.eval()
    
    all_probs = []
    all_true = []
    all_y_pred = []
    all_subject_ids = []
    all_paths = []

    for X, y, subject_ids, paths in loader:
        X = X.to(device)
        y = y.to(device)
        subject_ids = subject_ids

        logits = model(X)                # (B,2)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(stroke)
        preds = torch.argmax(logits, dim=1)

        all_probs.append(probs.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
        all_y_pred.append(preds.detach().cpu().numpy())
        all_subject_ids.extend(subject_ids)
        all_paths.extend(paths)

    y_true = np.concatenate(all_true).astype(int)
    y_score = np.concatenate(all_probs).astype(float)
    y_pred = np.concatenate(all_y_pred).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    bal_acc = 0.5 * (tpr + tnr)
    
    auc = roc_auc_score(y_true, y_score)
    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "bal_acc": bal_acc,
        "auc": auc,
        "stats": [tp, tn, fp, fn],
        "conf_mat": conf_mat,
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
        "subject_ids": all_subject_ids,
        "paths": paths
    }

@torch.no_grad()
def eval_subject_level(model, loader, threshold=0.5, device="cpu"):
    """Aggregate plot predictions to the subject level and evaluate them.

    Parameters
    ----------
    model : torch.nn.Module
        Fitted classification model.
    loader : torch.utils.data.DataLoader
        DataLoader yielding ``(X, y, subject_id, path)`` batches.
    threshold : float, default 0.5
        Decision threshold applied to the subject-level mean score.
    device : str or torch.device, default "cpu"
        Device used to move the batch tensors.

    Returns
    -------
    dict
        Subject-level scores, predictions, confusion matrix, and metrics.
    """
    model.eval()

    subject_scores = defaultdict(list)
    subject_labels = {}

    for X, y, subject_ids, _ in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)  # shape: (B, 2)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(class=1)
        y_np = y.cpu().numpy()

        for sid, label, score in zip(subject_ids, y_np, probs):
            subject_scores[sid].append(float(score))
            subject_labels[sid] = int(label)

    agg_subject_ids = []
    y_true = []
    y_score = []

    for sid in subject_scores:
        agg_subject_ids.append(sid)
        y_true.append(subject_labels[sid])
        y_score.append(np.mean(subject_scores[sid]))  # mean over that subject's plots

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = conf_mat.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (tpr + tnr)

    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else None

    return {
        "bal_acc": bal_acc,
        "auc": auc,
        "conf_mat": conf_mat,
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
        "subject_ids": agg_subject_ids,
    }


def evaluate_at_threshold(y_true, y_score, threshold):
    """Evaluate binary predictions at a fixed threshold.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_score : array-like
        Continuous scores or probabilities.
    threshold : float
        Decision threshold used to binarize ``y_score``.

    Returns
    -------
    dict
        Balanced accuracy, AUC, confusion matrix, and thresholded outputs.
    """
    y_pred = (y_score >= threshold).astype(int)

    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = conf_mat.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (tpr + tnr)

    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else None

    return {
        "bal_acc": bal_acc,
        "auc": auc,
        "stats": [tp, tn, fp, fn],
        "conf_mat": conf_mat,
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
        "threshold": threshold,
    }

