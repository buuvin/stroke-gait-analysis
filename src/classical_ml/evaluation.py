"""Model evaluation and visualization helpers for classical ML experiments."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
import matplotlib.pyplot as plt


def _predict_proba_safe(model, X):
    """Return class probabilities when the estimator supports them.

    Parameters
    ----------
    model : estimator
        Fitted estimator or pipeline.
    X : array-like
        Input samples for scoring.

    Returns
    -------
    numpy.ndarray or None
        Probability matrix when available, otherwise ``None``.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None


def _decision_function_safe(model, X):
    """Return decision scores when the estimator exposes them.

    Parameters
    ----------
    model : estimator
        Fitted estimator or pipeline.
    X : array-like
        Input samples for scoring.

    Returns
    -------
    numpy.ndarray or None
        Decision scores when available, otherwise ``None``.
    """
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            return None
    return None

def evaluate_binary_model(model, X_test, y_test, pos_label=1):
    """Compute binary classification metrics for a fitted model.

    Parameters
    ----------
    model : estimator
        Fitted binary classifier or pipeline.
    X_test : array-like
        Test features.
    y_test : array-like
        Binary labels encoded as 0/1.
    pos_label : int, default 1
        Label treated as the positive class for F1 computation.

    Returns
    -------
    dict
        Accuracy, ROC-AUC, sensitivity, specificity, F1, and confusion-matrix
        counts for the positive and negative classes.
    """
    y_test = np.asarray(y_test)
    y_pred = model.predict(X_test)

    # Confusion matrix in sklearn order: [[tn, fp], [fn, tp]] if labels=[0,1]
    labels = np.unique(y_test)
    if set(labels) != {0, 1}:
        raise ValueError(f"Binary y_test must be coded as 0/1. Got labels={labels}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = balanced_accuracy_score(y_test, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan  # TPR
    specificity = tn / (tn + fp) if (tn + fp) else np.nan  # TNR
    f1 = f1_score(y_test, y_pred, pos_label=pos_label)

    # ROC-AUC
    proba = _predict_proba_safe(model, X_test)
    if proba is not None and proba.shape[1] == 2:
        y_score = proba[:, 1]
    else:
        # fallback: decision_function
        dec = _decision_function_safe(model, X_test)
        if dec is None:
            y_score = None
        else:
            y_score = dec

    if y_score is not None:
        auc = roc_auc_score(y_test, y_score)
        fpr, tpr, thresh = roc_curve(y_test, y_score)
    else:
        auc = np.nan
        fpr, tpr, thresh = None, None, None

    report = classification_report(y_test, y_pred, digits=3, output_dict=True)

    out = {
        "accuracy": acc,
        "roc_auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_pos": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        # "report_dict": report,
        # "roc_curve": (fpr, tpr, thresh),
        # "confusion_matrix": cm
    }
    return out

def evaluate_models_binary(models: dict, X_test, y_test, task_name="binary"):
    """Evaluate several binary models on the same test set.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping from model name to fitted classifier.
    X_test : array-like
        Shared test features.
    y_test : array-like
        Shared binary test labels.
    task_name : str, default "binary"
        Task label added to the summary table.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, dict]]
        Tabular summary plus the per-model metric dictionaries.
    """
    rows = []
    details = {}

    for name, model in models.items():
        res = evaluate_binary_model(model, X_test, y_test)
        details[name] = res
        rows.append({
            "task": task_name,
            "model": name,
            "roc_auc": res["roc_auc"],
            "accuracy": res["accuracy"],
            "sensitivity": res["sensitivity"],
            "specificity": res["specificity"],
            "f1_pos": res["f1_pos"],
            "tn": res["tn"], "fp": res["fp"], "fn": res["fn"], "tp": res["tp"],
        })

    df = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)
    return df, details

def evaluate_multiclass_model(model, X_test, y_test, class_names=None, compute_ovr_auc=True):
    """Compute multiclass classification metrics for a fitted model.

    Parameters
    ----------
    model : estimator
        Fitted multiclass classifier or pipeline.
    X_test : array-like
        Test features.
    y_test : array-like
        Integer-encoded class labels.
    class_names : dict or None, default None
        Optional mapping from encoded labels to display names.
    compute_ovr_auc : bool, default True
        Whether to compute macro one-vs-rest ROC-AUC when probabilities are
        available.

    Returns
    -------
    dict
        Balanced accuracy, accuracy, macro F1, per-class recall, and OvR AUC.
    """
    y_test = np.asarray(y_test)
    y_pred = model.predict(X_test)

    labels = np.unique(y_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Per-class recall = diagonal / row sum
    recalls = {}
    for i, lab in enumerate(labels):
        denom = cm[i, :].sum()
        recalls[lab] = (cm[i, i] / denom) if denom else np.nan

    # friendly name mapping
    if class_names is None:
        class_names = {lab: str(lab) for lab in labels}

    per_class_recall_named = {class_names[lab]: recalls[lab] for lab in labels}

    report = classification_report(y_test, y_pred, digits=3, output_dict=True)

    ovr_auc = np.nan
    if compute_ovr_auc:
        proba = _predict_proba_safe(model, X_test)
        if proba is not None and proba.shape[1] == len(labels):
            Y_bin = label_binarize(y_test, classes=labels)
            ovr_auc = roc_auc_score(Y_bin, proba, average="macro", multi_class="ovr")

    out = {
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall_named,
        "ovr_roc_auc_macro": ovr_auc,
        #"report_dict": report,
        #"confusion_matrix": cm,
        # "labels": labels
    }
    return out

def evaluate_models_multiclass(models: dict, X_test, y_test, class_names=None, task_name="multiclass"):
    """Evaluate multiple multiclass models on the same test set.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping from model name to fitted classifier.
    X_test : array-like
        Shared test features.
    y_test : array-like
        Shared integer-encoded class labels.
    class_names : dict or None, default None
        Optional mapping from encoded labels to display names.
    task_name : str, default "multiclass"
        Task label added to the summary table.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, dict]]
        Tabular summary plus the per-model metric dictionaries.
    """
    rows = []
    details = {}

    for name, model in models.items():
        res = evaluate_multiclass_model(model, X_test, y_test, class_names=class_names)
        details[name] = res

        row = {
            "task": task_name,
            "model": name,
            "balanced_accuracy": res["balanced_accuracy"],
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
        }
        # add per-class recall into columns
        for cls, val in res["per_class_recall"].items():
            row[f"recall_{cls}"] = val

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(by="balanced_accuracy", ascending=False)
    return df, details


def plot_binary_roc_comparison(models, X_test, y_test, labels=None,
                            title="ROC Curve Comparison", save_path=None):
    """Plot ROC curves for multiple binary models on a single figure.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping from label to fitted binary classifier (pipeline).
    X_test, y_test : array-like
        Test data and 0/1 labels.
    labels : dict[str, str] | None
        Optional alternative display labels for the legend.
    title : str
        Title for the ROC comparison plot.
    save_path : str | None
        Optional path to save the figure.

    Returns
    -------
    dict[str, float]
        Mapping from model name to ROC AUC.

    Note
    ----
    This helper is binary-only. A future extension can add a `multi_class`
    option for macro/micro-averaged multiclass ROC curves.
    """
    y_test = np.asarray(y_test)

    results = {}
    plt.figure()

    for name, model in models.items():
        # Score: predict_proba if available, else decision_function.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba is None or proba.ndim != 2 or proba.shape[1] < 2:
                continue
            y_score = proba[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            # Skip models that cannot produce a continuous score.
            continue

        try:
            auc = roc_auc_score(y_test, y_score)
            fpr, tpr, _ = roc_curve(y_test, y_score)
        except Exception:
            # Skip models that fail ROC/AUC computation.
            continue

        display_name = labels.get(name, name) if labels is not None else name
        plt.plot(fpr, tpr, label=f"{display_name} (AUC = {auc:.3f})")
        results[name] = auc

    # Chance baseline
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return results

def plot_binary_roc_combined(model_eo, X_test_eo, y_test_eo, model_ec, X_test_ec, y_test_ec, title="XGB ROC Comparison", save_path=None):
    """Plot ROC curves for the eyes-open and eyes-closed XGB models.

    Parameters
    ----------
    model_eo, model_ec : fitted pipeline
        Fitted eyes-open and eyes-closed XGB pipelines.
    X_test_eo, y_test_eo : array-like
        Eyes-open test features and binary labels.
    X_test_ec, y_test_ec : array-like
        Eyes-closed test features and binary labels.
    title : str, default "XGB ROC Comparison"
        Title for the plot.
    save_path : str or None, default None
        Optional path to save the figure.

    Returns
    -------
    None
        The figure is displayed and optionally saved.
    """
    # Get scores
    y_score_eo = model_eo.predict_proba(X_test_eo)[:, 1]
    y_score_ec = model_ec.predict_proba(X_test_ec)[:, 1]
    
    # Compute ROC curves
    fpr_eo, tpr_eo, _ = roc_curve(y_test_eo, y_score_eo)
    auc_eo = roc_auc_score(y_test_eo, y_score_eo)
    
    fpr_ec, tpr_ec, _ = roc_curve(y_test_ec, y_score_ec)
    auc_ec = roc_auc_score(y_test_ec, y_score_ec)
    
    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_eo, tpr_eo, label=f"Eyes Open (AUC = {auc_eo:.3f})", linewidth=2)
    plt.plot(fpr_ec, tpr_ec, label=f"Eyes Closed (AUC = {auc_ec:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Eyes Open AUC: {auc_eo:.4f}")
    print(f"Eyes Closed AUC: {auc_ec:.4f}")


def plot_multiclass_roc(model, X_test, y_test, class_names, title="Model", save_path=None):
    """Plot one-vs-rest ROC curves for a multiclass model.

    Parameters
    ----------
    model : estimator
        Fitted multiclass classifier with ``predict_proba``.
    X_test : array-like
        Test features.
    y_test : array-like
        Integer-encoded labels in the range ``0..K-1``.
    class_names : list[str]
        Display names for the classes in encoded order.
    title : str, default "Model"
        Plot title suffix.
    save_path : str or None, default None
        Optional path to save the figure.

    Returns
    -------
    float
        Macro one-vs-rest ROC-AUC.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Multiclass ROC needs predict_proba().")

    y_proba = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

    auc_macro = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")

    plt.figure()
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_i:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multiclass ROC (OvR) — {title}\nMacro AUC = {auc_macro:.3f}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return auc_macro

def run_cv_scores(pipelines, X, y, cv_splits=5):
    """Run cross-validated accuracy scoring for each pipeline.

    Parameters
    ----------
    pipelines : dict[str, estimator]
        Mapping from model name to pipeline.
    X : array-like
        Feature matrix.
    y : array-like
        Target labels.
    cv_splits : int, default 5
        Number of stratified group folds.

    Returns
    -------
    dict[str, dict]
        Per-model mean, standard deviation, and fold-level scores.
    """
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(
            pipe,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        results[name] = {
            "mean": scores.mean(),
            "std": scores.std(),
            "all_scores": scores
        }
        print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    return results


def plot_cv_boxplots_with_points(cv_scores_eo, cv_scores_ec, save_path=None, jitter=0.06):
    """Plot cross-validation score distributions with jittered fold points.

    Parameters
    ----------
    cv_scores_eo, cv_scores_ec : array-like
        Cross-validation scores for the two conditions.
    save_path : str or None, default None
        Optional path to save the figure.
    jitter : float, default 0.06
        Horizontal jitter applied to the fold points.

    Returns
    -------
    None
        The figure is displayed and optionally saved.
    """
    cv_scores_eo = np.asarray(cv_scores_eo, dtype=float)
    cv_scores_ec = np.asarray(cv_scores_ec, dtype=float)

    plt.figure()
    plt.boxplot(
        [cv_scores_eo, cv_scores_ec],
        labels=["Eyes Open", "Eyes Closed"],
        showmeans=True
    )

    # overlay points jitter
    x1 = 1 + np.random.uniform(-jitter, jitter, size=len(cv_scores_eo))
    x2 = 2 + np.random.uniform(-jitter, jitter, size=len(cv_scores_ec))
    plt.scatter(x1, cv_scores_eo, marker="o")
    plt.scatter(x2, cv_scores_ec, marker="o")

    plt.ylabel("Cross-Validation Accuracy")
    plt.title("CV Scores (XGB) — Eyes Open vs Eyes Closed")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()