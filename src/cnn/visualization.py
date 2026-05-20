"""Visualization helpers for CNN-based RQA classification."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

def plot_roc_curve(eval_out):
    """Plot the ROC curve for one evaluation output.

    Parameters
    ----------
    eval_out : dict
        Evaluation dictionary containing ``y_true`` and ``y_score``.

    Returns
    -------
    None
        The figure is displayed.
    """
    y_true = eval_out["y_true"]
    y_score = eval_out["y_score"]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_roc_curve_cv(fold_results):
    """Plot a mean ROC curve across cross-validation folds.

    Parameters
    ----------
    fold_results : list[dict]
        Fold dictionaries returned by :func:`cnn.training.run_subject_kfold_cv`.

    Returns
    -------
    None
        The figure is displayed.
    """
    fold_fprs = []
    fold_tprs = []
    fold_aucs = []
    
    for fold_dict in fold_results:
        best_eval_out = fold_dict.get("best_eval_out")
        if best_eval_out is None:
            print(f"Warning: Fold {fold_dict.get('fold')} missing best_eval_out")
            continue
        
        y_true = best_eval_out["y_true"]
        y_score = best_eval_out["y_score"]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fold_auc = auc(fpr, tpr)
        
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)
        fold_aucs.append(fold_auc)
    
    if not fold_fprs:
        print("No fold data available to plot")
        return
    
    fpr_mean = np.linspace(0, 1, 100)
    
    tpr_interp = []
    for fpr, tpr in zip(fold_fprs, fold_tprs):
        tpr_interp.append(np.interp(fpr_mean, fpr, tpr))
    
    tpr_interp = np.array(tpr_interp)
    tpr_mean = tpr_interp.mean(axis=0)
    tpr_std = tpr_interp.std(axis=0)
    
    all_y_true = []
    all_y_score = []
    for fold_dict in fold_results:
        best_eval_out = fold_dict.get("best_eval_out")
        if best_eval_out is not None:
            all_y_true.append(best_eval_out["y_true"])
            all_y_score.append(best_eval_out["y_score"])
    
    y_true_all = np.concatenate(all_y_true)
    y_score_all = np.concatenate(all_y_score)
    fpr_all, tpr_all, _ = roc_curve(y_true_all, y_score_all)
    roc_auc_all = auc(fpr_all, tpr_all)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_mean, tpr_mean, label=f"Mean CV AUC = {roc_auc_all:.3f}", linewidth=2)
    plt.fill_between(fpr_mean, 
                      tpr_mean - tpr_std, 
                      tpr_mean + tpr_std, 
                      alpha=0.2, 
                      label="± 1 Std Dev")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Cross-Validation, All Folds)")
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()
    
    print(f"Combined CV ROC AUC: {roc_auc_all:.4f}")
    print(f"Individual fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

def plot_conf_matrix(eval_out, class_names=("Healthy", "Stroke")):
    """Plot a confusion matrix from an evaluation output dictionary.

    Parameters
    ----------
    eval_out : dict
        Evaluation dictionary containing ``conf_mat``.
    class_names : tuple[str, str], default ("Healthy", "Stroke")
        Display labels for the two classes.

    Returns
    -------
    None
        The figure is displayed.
    """
    conf_mat = eval_out["conf_mat"]

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()