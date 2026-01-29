import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
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
        # Initialize models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "SVC": SVC(probability=True, random_state=random_state),
            "K-Nearest Neighbors": KNeighborsClassifier()
        }
        
        metrics = {}
        trained_models = {}
        best_model_name = None
        best_score = -1
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average="macro")
                
                metrics[f"{name}_accuracy"] = float(acc)
                metrics[f"{name}_f1_macro"] = float(f1)
                trained_models[name] = model
                
                if acc > best_score:
                    best_score = acc
                    best_model_name = name
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        results["metrics"] = metrics
        results["models"] = trained_models
        results["best_model"] = best_model_name
        results["best_model_score"] = best_score
        results["feature_names"] = X.columns.tolist()
        
        # Feature importance (try RF first, then GB, then DT)
        for m_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            if m_name in trained_models and hasattr(trained_models[m_name], "feature_importances_"):
                importances = pd.Series(trained_models[m_name].feature_importances_, index=X.columns).sort_values(ascending=False)
                results["feature_importance"] = importances.reset_index().rename(columns={"index": "feature", 0: "importance"})
                break
    else:
        # Initialize models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=random_state),
            "Lasso Regression": Lasso(random_state=random_state),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state)
        }
        
        # Train and evaluate
        metrics = {}
        trained_models = {}
        best_model_name = None
        best_score = -float('inf')
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                r2 = r2_score(y_test, pred)
                mse = mean_squared_error(y_test, pred)
                
                metrics[f"{name}_r2"] = float(r2)
                metrics[f"{name}_mse"] = float(mse)
                trained_models[name] = model
                
                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        results["metrics"] = metrics
        results["models"] = trained_models
        results["best_model"] = best_model_name
        results["best_model_score"] = best_score
        results["feature_names"] = X.columns.tolist()
        
        # Feature importance
        for m_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            if m_name in trained_models and hasattr(trained_models[m_name], "feature_importances_"):
                importances = pd.Series(trained_models[m_name].feature_importances_, index=X.columns).sort_values(ascending=False)
                results["feature_importance"] = importances.reset_index().rename(columns={"index": "feature", 0: "importance"})
                break
    return results
