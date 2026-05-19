"""Train and evaluate classical ML models for eyes-open, eyes-closed, and limb tasks.

The script loads the engineered feature table, applies the subject-level split,
and runs the binary and multiclass pipelines used in the stroke gait analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedGroupKFold

from classical_ml.preprocessing import clean_data, preprocess_limb_data, preprocess_eyes_data
from classical_ml.models import pipelines, xgb_param_grid, cv
from classical_ml.train_test_split import make_subject_split, apply_subject_split, collect_train_test
from classical_ml.evaluation import evaluate_binary_model, plot_binary_roc_comparison, plot_binary_roc_combined, plot_multiclass_roc, plot_cv_boxplots_side_by_side
from classical_ml.feature_importances import shap_like_bar_from_pipeline_pred_contribs, xgb_gain_from_pipeline, plot_gain_bar
from paths import (FIGURES, RESULTS, ML_FEATURES_FILE, SUBJECT_SPLIT_FILE, EO_MODELS_RESULTS_FILE, EC_MODELS_RESULTS_FILE, LIMB_MODELS_RESULTS_FILE, 
                   LIMB_XGB_RESULTS_FILE, EYES_ROC_COMPARISON_PLOT, LIMB_ROC_COMPARISON_PLOT, EO_SHAP_PLOT, EC_SHAP_PLOT, LIMB_SHAP_PLOT, EO_GAIN_PLOT, 
                   EC_GAIN_PLOT, LIMB_CM_PLOT, EYES_CV_BOXPLOT)
from config import RANDOM
rand.seed(RANDOM)

def run_models(pipelines, X_train, X_test, y_train, y_test, out_file, task):
    """Fit each pipeline in ``pipelines`` and record its test-set metrics.

    Parameters
    ----------
    pipelines : dict[str, sklearn.pipeline.Pipeline]
        Mapping from model name to an unfitted pipeline.
    X_train, X_test : pandas.DataFrame
        Feature matrices for training and evaluation.
    y_train, y_test : array-like
        Training and test labels.
    out_file : pathlib.Path or str
        File where summary metrics are appended.
    task : str
        Label written to the results file for the current analysis block.

    Returns
    -------
    dict[str, sklearn.pipeline.Pipeline]
        The fitted pipelines keyed by model name.
    """
    models = {}
    
    with open(out_file, 'a') as f:
        f.write("\n=== " + task + " ===")
        for name, pipe in pipelines.items():
            pipe.fit(X_train, y_train)
            models[name] = pipe
            preds = pipe.predict(X_test)
            f.write("\n===", name, "===")
            f.write("Accuracy: " + str(accuracy_score(y_test, preds)))
            f.write(str(confusion_matrix(y_test, preds)))
            f.write(str(classification_report(y_test, preds)))
            f.write(str(evaluate_binary_model(pipe, X_test, y_test)))

    return models


def eyes_models_pipeline(df_eyes, cond, out_file):
    """Run the eyes-condition pipelines and XGBoost grid search.

    Parameters
    ----------
    df_eyes : pandas.DataFrame
        Eyes-open or eyes-closed feature table with subject split metadata.
    cond : str
        Condition label used in filenames and output titles.
    out_file : pathlib.Path or str
        File where per-model summary results are appended.

    Returns
    -------
    tuple
        ``(best_xgb, y_pred, [X_train, X_test, y_train, y_test])`` for the
        selected XGBoost model and the split used to evaluate it.
    """
    df_train = df_eyes[df_eyes["split"] == "train"]

    X_train, X_test, y_train, y_test = collect_train_test(df_eyes, cond)

    eyes_models = run_models(pipelines, X_train, X_test, y_train, y_test, out_file, cond)
    roc_auc_by_model = plot_binary_roc_comparison(
        models=eyes_models,          # same dict you passed to run_models
        X_test=X_test,
        y_test=y_test,
        title="ROC Curve Comparison — " + cond.replace("_", " ").title() + " (run_models pipelines)",
        save_path=FIGURES/"roc_compare_" + cond.replace("_", "") + "_all.png",
    )

    print("ROC AUC by model (EO):")
    for name, auc in roc_auc_by_model.items():
        print(f"  {name}: {auc:.3f}")

    xgb_grid = GridSearchCV(
    pipelines['xgb'],
    xgb_param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    refit = True,
    verbose = 1,
    error_score='raise'
    )

    xgb_grid.fit(X_train, y_train, groups=df_train["subject_id"].values)
    print("Best params:", xgb_grid.best_params_)
    print("Best CV score:", xgb_grid.best_score_)

    best_xgb = xgb_grid.best_estimator_
    best_xgb = xgb_grid.best_estimator_
    y_pred = best_xgb.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp =  ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=best_xgb.classes_)
    disp.plot()
    plt.savefig(FIGURES/ cond.replace("_", "") + "_cm.png", dpi=300, bbox_inches="tight")
    plt.show()
    with open(RESULTS/ cond.replace("_", "") + "_xgb_grid_results.csv", 'a') as f:
        f.write("\n=== " + cond + " ===")
        f.write("Best params: " + str(xgb_grid.best_params_))
        f.write("Best CV score: " + str(xgb_grid.best_score_))
        f.write(str(classification_report(y_test, y_pred)))
        f.write(str(evaluate_binary_model(best_xgb, X_test, y_test)))

    return best_xgb, y_pred, [X_train, X_test, y_train, y_test]

def eyes_classification(df_eyes_open, df_eyes_closed):
    """Run the full eyes-open versus eyes-closed classification workflow.

    Parameters
    ----------
    df_eyes_open : pandas.DataFrame
        Eyes-open feature table.
    df_eyes_closed : pandas.DataFrame
        Eyes-closed feature table.

    Returns
    -------
    None
        Figures and result tables are written to disk.
    """
    df_train_o = df_eyes_open[df_eyes_open["split"] == "train"]
    df_train_c = df_eyes_closed[df_eyes_closed["split"] == "train"]

    best_xgb_o, y_pred_o, train_test_split_eo = eyes_models_pipeline(df_eyes_open, "eyes_open", EO_MODELS_RESULTS_FILE)
    best_xgb_c, y_pred_c, train_test_split_ec = eyes_models_pipeline(df_eyes_closed, "eyes_closed", EC_MODELS_RESULTS_FILE)

    plot_binary_roc_combined(
        model_eo=best_xgb_o,
        X_test_eo=train_test_split_eo[1],
        y_test_eo=train_test_split_eo[3],
        model_ec=best_xgb_c,
        X_test_ec=train_test_split_ec[1],
        y_test_ec=train_test_split_ec[3],
        title="XGB ROC Comparison — Eyes Open vs Eyes Closed",
        save_path=EYES_ROC_COMPARISON_PLOT
    )

    X_ref_eo = train_test_split_eo[1].copy()
    if len(X_ref_eo) > 2000:
        X_ref_eo = X_ref_eo.sample(2000, random_state=42)

    eo_shap_like = shap_like_bar_from_pipeline_pred_contribs(
        best_xgb_o, X_ref_eo, title="SHAP-like Importance — Eyes Open", save_path=EO_SHAP_PLOT
    )

    eo_gain = xgb_gain_from_pipeline(best_xgb_o, X_ref=X_ref_eo, top_k=20)

    # Store max for later comparison with Eyes Closed (plots both together below)
    eo_gain_max = eo_gain.max() if len(eo_gain) > 0 else 0

    X_ref_ec = train_test_split_ec[1].copy()
    if len(X_ref_ec) > 2000:
        X_ref_ec = X_ref_ec.sample(2000, random_state=42)

    ec_shap_like = shap_like_bar_from_pipeline_pred_contribs(
        best_xgb_c, X_ref_ec, title="SHAP-like Importance — Eyes Closed", save_path=EC_SHAP_PLOT
    )

    ec_gain = xgb_gain_from_pipeline(best_xgb_c, X_ref=X_ref_ec, top_k=20)

    ec_gain = xgb_gain_from_pipeline(best_xgb_c, X_ref=X_ref_ec, top_k=20)

    # Compute shared xlim from both Eyes Open and Eyes Closed
    shared_xlim = max(eo_gain_max, ec_gain.max() if len(ec_gain) > 0 else 0) * 1.1  # Add 10% padding

    # Plot both with shared x-axis
    plot_gain_bar(eo_gain, "XGBoost Gain Importance — Eyes Open", EO_GAIN_PLOT, xlim=shared_xlim)
    plot_gain_bar(ec_gain, "XGBoost Gain Importance — Eyes Closed", EC_GAIN_PLOT, xlim=shared_xlim)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    groups_o = df_train_o["subject_id"].values
        
    scores_o = cross_val_score(
        best_xgb_o,
        train_test_split_eo[0],
        train_test_split_eo[2],
        groups=groups_o,
        cv=cv,
        scoring='balanced_accuracy'
    )

    groups_c = df_train_c["subject_id"].values

    scores_c = cross_val_score(
        best_xgb_c,
        train_test_split_ec[0],
        train_test_split_ec[2],
        groups=groups_c,
        cv=cv,
        scoring='balanced_accuracy'
    )

    plot_cv_boxplots_side_by_side(scores_o, scores_c, save_path=EYES_CV_BOXPLOT)

    return

def limb_classification(df_limb):
    """Run the multiclass limb-status classification workflow.

    Parameters
    ----------
    df_limb : pandas.DataFrame
        Limb-level feature table with ``limb_status`` labels and subject splits.

    Returns
    -------
    None
        Figures and result tables are written to disk.
    """
    df_train = df_limb[df_limb["split"] == "train"]
    df_test = df_limb[df_limb["split"] == "test"]
    
    df_train = df_limb[df_limb["split"] == "train"]

    X_train, X_test, y_train, y_test = collect_train_test(df_limb, "limb")

    limb_models = run_models(pipelines, X_train, X_test, y_train, y_test, LIMB_MODELS_RESULTS_FILE, "limb")
    roc_auc_by_model = plot_multiclass_roc(
        models=limb_models,          # same dict you passed to run_models
        X_test=X_test,
        y_test=y_test,
        title="ROC Curve Comparison — " + "Limb" + " (run_models pipelines)",
        save_path=LIMB_ROC_COMPARISON_PLOT,
    )

    print("ROC AUC by model (Limb):")
    for name, auc in roc_auc_by_model.items():
        print(f"  {name}: {auc:.3f}")

    xgb_grid = GridSearchCV(
    pipelines['xgb'],
    xgb_param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    refit = True,
    verbose = 1,
    error_score='raise'
    )

    xgb_grid.fit(X_train, y_train, groups=df_train["subject_id"].values)
    print("Best params:", xgb_grid.best_params_)
    print("Best CV score:", xgb_grid.best_score_)

    best_xgb = xgb_grid.best_estimator_
    y_pred = best_xgb.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp =  ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=best_xgb.classes_)
    disp.plot()
    plt.savefig(LIMB_CM_PLOT, dpi=300, bbox_inches="tight")
    plt.show()
    with open(LIMB_XGB_RESULTS_FILE, 'a') as f:
        f.write("\n=== " + "limb" + " ===")
        f.write("Best params: " + str(xgb_grid.best_params_))
        f.write("Best CV score: " + str(xgb_grid.best_score_))
        f.write(str(classification_report(y_test, y_pred)))
        f.write(str(evaluate_binary_model(best_xgb, X_test, y_test)))

    X_ref = X_test.copy()
    if len(X_ref) > 2000:
        X_ref = X_ref.sample(2000, random_state=42)

    ec_shap_like = shap_like_bar_from_pipeline_pred_contribs(
        best_xgb, X_ref, title="SHAP-like Importance — Limb", save_path=LIMB_SHAP_PLOT
    )

    
    return

def main():
    """Load features, prepare splits, and run all classification workflows.

    Returns
    -------
    None
        All outputs are persisted to the configured paths.
    """
    features_raw = pd.read_csv(ML_FEATURES_FILE)
    subject_split = make_subject_split(features_raw, out_csv = SUBJECT_SPLIT_FILE)
    df_split = apply_subject_split(features_raw, subject_split)
    df_clean = clean_data(df_split)

    df_eyes_open, df_eyes_closed = preprocess_eyes_data(df_clean)
    df_limb = preprocess_limb_data(df_clean)

    eyes_classification(df_eyes_open, df_eyes_closed)

    limb_classification(df_limb)