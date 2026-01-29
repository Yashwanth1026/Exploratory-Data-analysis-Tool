import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

def determine_problem_type(y):
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= 20:
            return "classification"
        return "regression"
    return "classification"

def prepare_features(df, target):
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.fillna(X.mean())
    return X, y

def train_and_evaluate(df, target, test_size=0.2, random_state=42):
    X, y = prepare_features(df, target)
    problem = determine_problem_type(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results = {"problem_type": problem}
    if problem == "classification":
        lr = LogisticRegression(max_iter=1000)
        rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        rf_pred = rf.predict(X_test)
        results["metrics"] = {
            "log_reg_accuracy": float(accuracy_score(y_test, lr_pred)),
            "log_reg_f1_macro": float(f1_score(y_test, lr_pred, average="macro")),
            "rf_accuracy": float(accuracy_score(y_test, rf_pred)),
            "rf_f1_macro": float(f1_score(y_test, rf_pred, average="macro")),
        }
        if hasattr(rf, "feature_importances_"):
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            results["feature_importance"] = importances.reset_index().rename(columns={"index": "feature", 0: "importance"})
    else:
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        rf_pred = rf.predict(X_test)
        results["metrics"] = {
            "lin_reg_r2": float(r2_score(y_test, lr_pred)),
            "lin_reg_mse": float(mean_squared_error(y_test, lr_pred)),
            "rf_r2": float(r2_score(y_test, rf_pred)),
            "rf_mse": float(mean_squared_error(y_test, rf_pred)),
        }
        if hasattr(rf, "feature_importances_"):
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            results["feature_importance"] = importances.reset_index().rename(columns={"index": "feature", 0: "importance"})
    return results
