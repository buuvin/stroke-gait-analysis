"""Pipeline and hyperparameter definitions for the classical ML models."""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

eyes_pipelines = {
    "logreg": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ]),
    "knn": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "dt": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(random_state=42))
    ]),
    "svm": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ]),
    "rf": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("rf", RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ]),
    "xgb": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        ))
    ])
}

limb_pipelines = {
    'log' : Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=5000,
            class_weight='balanced'
        ))
    ]),
    'tree' : Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', DecisionTreeClassifier(
            class_weight='balanced',
            random_state=42
        ))
    ]),
    'knn' : Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=7))
    ]),
    "svm": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ]),
    "rf": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("rf", RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ]),
    'xg_boost' : Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('clf', XGBClassifier(
            objective='multi:softprob',
            random_state=42,
        ))
    ])
}

xbg_param_grid = {
    'clf__n_estimators': [300, 500, 800],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.03, 0.05, 0.1],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0],
    'clf__reg_lambda': [1.0, 5.0]
}