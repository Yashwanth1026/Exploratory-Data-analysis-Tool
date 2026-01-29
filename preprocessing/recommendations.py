import pandas as pd
import numpy as np

def generate_recommendations(df):
    """
    Generates structured, user-friendly recommendations for data preprocessing.
    Returns a list of dictionaries with keys: category, column, severity, message, suggestion.
    """
    recommendations = []
    
    if df is None or df.empty:
        return recommendations

    # 1. Missing Values
    missing_pct = df.isnull().mean() * 100
    for col, pct in missing_pct.items():
        if pct > 0:
            rec = {
                "category": "Missing Data",
                "column": col,
                "value": f"{pct:.1f}%",
                "severity": "High" if pct > 50 else "Medium",
                "message": f"Column '{col}' is missing {pct:.1f}% of its data.",
                "suggestion": "Drop column" if pct > 50 else "Impute (fill) values" if pct > 5 else "Drop rows"
            }
            if pct > 50:
                rec["explanation"] = "More than half the data is missing. Keeping this column might introduce noise."
            elif pct > 5:
                rec["explanation"] = "A significant portion is missing. Filling gaps (imputation) allows you to keep the data."
            else:
                rec["explanation"] = "Only a few rows are missing. Removing them is usually safe."
            recommendations.append(rec)

    # 2. Skewness (Numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        skewness = df[numeric_cols].skew(numeric_only=True)
        for col, skew in skewness.items():
            if abs(skew) > 1.0:
                recommendations.append({
                    "category": "Distribution",
                    "column": col,
                    "value": f"{skew:.2f} skew",
                    "severity": "Medium",
                    "message": f"Column '{col}' is skewed ({skew:.2f}).",
                    "suggestion": "Apply Log/Box-Cox Transform",
                    "explanation": "The data is not symmetrical. Transformations can make it more 'normal' for models."
                })

    # 3. Outliers (IQR)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            pct_outliers = (outliers / len(df)) * 100
            if pct_outliers > 1: # Only report if > 1%
                 recommendations.append({
                    "category": "Outliers",
                    "column": col,
                    "value": f"{pct_outliers:.1f}%",
                    "severity": "High" if pct_outliers > 10 else "Medium",
                    "message": f"Column '{col}' has {outliers} outliers ({pct_outliers:.1f}%).",
                    "suggestion": "Handle Outliers (Clip/Remove)",
                    "explanation": "Extreme values can distort analysis and models."
                })

    # 4. High Correlation
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        for col in to_drop:
             recommendations.append({
                "category": "Correlation",
                "column": col,
                "value": "> 0.9",
                "severity": "Low",
                "message": f"Column '{col}' is highly correlated with others.",
                "suggestion": "Remove Feature",
                "explanation": "This information is redundant as it's already captured by other columns."
            })

    # 5. Categorical High Cardinality
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count > 50:
             recommendations.append({
                "category": "Cardinality",
                "column": col,
                "value": f"{unique_count} unique",
                "severity": "Medium",
                "message": f"Column '{col}' has many unique values ({unique_count}).",
                "suggestion": "Frequency Encoding / Top-N",
                "explanation": "Too many categories can confuse models. Try grouping rare ones."
            })
    
    # 6. Constant Columns
    for col in df.columns:
        if df[col].nunique() <= 1:
             recommendations.append({
                "category": "Redundant",
                "column": col,
                "value": "Constant",
                "severity": "High",
                "message": f"Column '{col}' has only 1 unique value.",
                "suggestion": "Drop Column",
                "explanation": "This column offers no information since it's the same for every row."
            })

    return recommendations
