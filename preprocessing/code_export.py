import pandas as pd

def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def generate_sklearn_pipeline_code(df, numeric_cols=None, categorical_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    code = []
    code.append("from sklearn.pipeline import Pipeline")
    code.append("from sklearn.compose import ColumnTransformer")
    code.append("from sklearn.preprocessing import OneHotEncoder, StandardScaler")
    code.append("from sklearn.impute import SimpleImputer")
    code.append("")
    code.append(f"numeric_features = {numeric_cols}")
    code.append(f"categorical_features = {categorical_cols}")
    code.append("")
    code.append("numeric_transformer = Pipeline(steps=[")
    code.append("    ('imputer', SimpleImputer(strategy='median')),")
    code.append("    ('scaler', StandardScaler())")
    code.append("])")
    code.append("")
    code.append("categorical_transformer = Pipeline(steps=[")
    code.append("    ('imputer', SimpleImputer(strategy='most_frequent')),")
    code.append("    ('encoder', OneHotEncoder(handle_unknown='ignore'))")
    code.append("])")
    code.append("")
    code.append("preprocessor = ColumnTransformer(")
    code.append("    transformers=[")
    code.append("        ('num', numeric_transformer, numeric_features),")
    code.append("        ('cat', categorical_transformer, categorical_features)")
    code.append("    ]")
    code.append(")")
    code.append("")
    code.append("model_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])")
    code.append("")
    return "\n".join(code)
