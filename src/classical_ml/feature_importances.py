"""Feature importance helpers for fitted classical ML pipelines."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.pipeline import Pipeline

def shap_like_bar_from_pipeline_pred_contribs(pipe, X_ref, title, max_display=20, save_path=None):
    """Compute SHAP-like importances from XGBoost prediction contributions.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline or estimator
        Fitted pipeline or fitted XGBoost estimator.
    X_ref : pandas.DataFrame or array-like
        Reference data used to compute contribution magnitudes.
    title : str
        Figure title.
    max_display : int, default 20
        Maximum number of features to show in the bar plot.
    save_path : str or None, default None
        Optional output path for the saved figure.

    Returns
    -------
    pandas.Series
        Mean absolute contribution per feature, sorted descending.
    """

    # Split pipeline into preprocess + model
    if isinstance(pipe, Pipeline):
        model = pipe.steps[-1][1]
        pre = pipe[:-1] if len(pipe.steps) > 1 else None
    else:
        # not a pipeline
        model = pipe
        pre = None

    # Transform X_ref if preprocessing exists
    if pre is not None:
        X_trans = pre.transform(X_ref)
        try:
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = [f"f{i}" for i in range(X_trans.shape[1])]
    else:
        X_trans = X_ref.values if hasattr(X_ref, "values") else np.asarray(X_ref)
        feat_names = list(X_ref.columns) if hasattr(X_ref, "columns") else [f"f{i}" for i in range(X_trans.shape[1])]

    # Make sure it's dense
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # Get booster
    booster = model.get_booster() if hasattr(model, "get_booster") else model

    dmat = xgb.DMatrix(X_trans, feature_names=list(feat_names))
    contrib = booster.predict(dmat, pred_contribs=True)

    # Handle binary vs multiclass shapes
    if contrib.ndim == 3:
        # (n_samples, n_classes, n_features+1) -> avg abs over classes
        contrib_abs = np.mean(np.abs(contrib[:, :, :-1]), axis=1)
    else:
        # (n_samples, n_features+1)
        contrib_abs = np.abs(contrib[:, :-1])

    mean_abs = contrib_abs.mean(axis=0)
    s = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)

    plt.figure()
    s.head(max_display).sort_values().plot(kind="barh")
    plt.xlabel("mean(|contribution|)  (XGBoost pred_contribs)")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return s

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def xgb_gain_from_pipeline(best_estimator, X_ref=None, top_k=20):
    """Extract XGBoost gain importances from a fitted pipeline.

    Parameters
    ----------
    best_estimator : sklearn.pipeline.Pipeline or estimator
        Fitted pipeline or fitted XGBoost estimator.
    X_ref : pandas.DataFrame or None, default None
        Optional reference data used to recover transformed feature names.
    top_k : int, default 20
        Number of top features to return.

    Returns
    -------
    pandas.Series
        Gain importance values indexed by feature name when available.
    """
    # Unwrap pipeline if needed
    if isinstance(best_estimator, Pipeline):
        model = best_estimator.steps[-1][1]
        pre = best_estimator[:-1] if len(best_estimator.steps) > 1 else None
    else:
        model = best_estimator
        pre = None

    booster = model.get_booster()

    # Get raw gain scores keyed like "f0", "f1", ...
    score = booster.get_score(importance_type="gain")
    if len(score) == 0:
        # sometimes happens if model hasn't been fit or booster is empty
        return pd.Series(dtype=float)

    # Determine feature names in the model's input space
    feat_names = None

    # If we have a preprocessor, try to use its output names
    if pre is not None:
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            # fallback: infer dimensionality from a transform on X_ref if provided
            if X_ref is not None:
                Xt = pre.transform(X_ref)
                n_feat = Xt.shape[1]
                feat_names = [f"f{i}" for i in range(n_feat)]
            else:
                feat_names = None

    # If no preprocessor, use X_ref columns if possible
    if feat_names is None and X_ref is not None and hasattr(X_ref, "columns"):
        feat_names = list(X_ref.columns)

    # Map f0..fN to names if we have them
    if feat_names is not None:
        fmap = {f"f{i}": name for i, name in enumerate(feat_names)}
        imp = {fmap.get(k, k): v for k, v in score.items()}
    else:
        imp = score  # keep f0,f1,...

    s = pd.Series(imp).sort_values(ascending=False)
    return s.head(top_k)

def plot_gain_bar(series, title, save_path=None, xlim=None):
    """Plot a horizontal bar chart for XGBoost gain importances.

    Parameters
    ----------
    series : pandas.Series
        Feature importance values indexed by feature name.
    title : str
        Figure title.
    save_path : str or None, default None
        Optional output path for the saved figure.
    xlim : float or None, default None
        Optional shared x-axis upper bound.

    Returns
    -------
    None
        The figure is displayed and optionally written to disk.
    """
    if series is None or len(series) == 0:
        print("No gain importances found (empty series).")
        return

    plt.figure()
    series.sort_values().plot(kind="barh")
    plt.xlabel("Importance (gain)")
    plt.title(title)
    if xlim is not None:
        plt.xlim([0, xlim])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()