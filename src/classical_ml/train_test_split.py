"""Subject-wise splitting helpers for the classical ML workflows."""

from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def make_subject_split(df, subject_col="subject_id", label_col="label",
                       test_size=0.2, random_state=42, out_csv="subject_split.csv"):
    """Create a reusable subject-wise train/test split.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing one or more rows per subject.
    subject_col : str, default "subject_id"
        Column containing the subject identifier.
    label_col : str, default "label"
        Subject-level label used to preserve class balance.
    test_size : float, default 0.2
        Fraction of subjects assigned to the test split within each class.
    random_state : int, default 42
        Seed used for the group shuffle split.
    out_csv : str, default "subject_split.csv"
        Path where the split table is written.

    Returns
    -------
    pandas.DataFrame
        Subject-to-split mapping with one row per subject.
    """
    df = df.copy()

    subj_label_counts = df.groupby(subject_col)[label_col].nunique()
    bad = subj_label_counts[subj_label_counts > 1]
    if len(bad) > 0:
        raise ValueError(f"Some subjects have multiple labels in `{label_col}`. Examples: {bad.head()}")

    subj_df = df.groupby(subject_col, as_index=False)[label_col].first()

    train_subjects = []
    test_subjects = []

    for lab, g in subj_df.groupby(label_col):
        subjects = g[subject_col].values

        if len(subjects) < 2:
            train_subjects.extend(subjects.tolist())
            continue

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        idx_train, idx_test = next(gss.split(subjects, groups=subjects))
        train_subjects.extend(subjects[idx_train].tolist())
        test_subjects.extend(subjects[idx_test].tolist())

    split_df = pd.DataFrame({
        subject_col: train_subjects + test_subjects,
        "split": (["train"] * len(train_subjects)) + (["test"] * len(test_subjects))
    }).drop_duplicates(subset=[subject_col])

    split_df.to_csv(out_csv, index=False)
    print(f"Saved subject-wise split to {out_csv} "
          f"(train subjects={sum(split_df.split=='train')}, test subjects={sum(split_df.split=='test')})")

    return split_df


def apply_subject_split(df, split_df, subject_col="subject_id"):
    """Attach the saved split assignment to each row of a feature table.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature table with one or more rows per subject.
    split_df : pandas.DataFrame
        Subject-to-split mapping created by :func:`make_subject_split`.
    subject_col : str, default "subject_id"
        Column containing the subject identifier.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with a new ``split`` column.
    """
    df = df.copy()
    m = dict(zip(split_df[subject_col], split_df["split"]))
    df["split"] = df[subject_col].map(m)

    missing = df["split"].isna()
    if missing.any():
        missing_subs = df.loc[missing, subject_col].unique()[:10]
        raise ValueError(f"Some subjects in df were not found in split_df. Examples: {missing_subs}")

    return df

def collect_train_test(data_df, target_col):
    """Build feature matrices and encoded labels for train/test evaluation.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Feature table with a ``split`` column.
    target_col : str
        Name of the prediction target column.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test)`` with labels integer encoded.
    """
    data_df['avg_diag_line_nan'] = data_df['avg_diag_line'].isna().astype(int)

    df_train = data_df[data_df["split"] == "train"]
    df_test = data_df[data_df["split"] == "test"]

    drop_cols = ['Unnamed: 0', 'filename', 'subject_id', 'label', 'category', 'eye_condition',
                'affected_side', 'cop_type', 'axis', 'split', target_col]

    feature_cols = [c for c in data_df.columns if c not in drop_cols]


    X_train, y_train = get_xy(df_train, feature_cols, target_col=target_col)
    X_test,  y_test  = get_xy(df_test, feature_cols, target_col=target_col)

    le_open = LabelEncoder()

    y_train = le_open.fit_transform(df_train[target_col])
    y_test = le_open.transform(df_test[target_col])

    return X_train, y_train, X_test, y_test

def get_xy(df, feature_cols, target_col):
    """Split a dataframe into feature matrix and target vector.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing the feature and target columns.
    feature_cols : list[str]
        Columns to use as model features.
    target_col : str
        Name of the target column.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Feature matrix ``X`` and target vector ``y``.
    """
    X = df[feature_cols]
    y = df[target_col]
    return X, y