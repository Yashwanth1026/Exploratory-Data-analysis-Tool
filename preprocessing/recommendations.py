import pandas as pd
import numpy as np

def generate_recommendations(df):
    """
    Generates simple text-based recommendations for data preprocessing.
    """
    recommendations = []
    
    if df is None or df.empty:
        return recommendations

    # 1. Missing Values
    missing_pct = df.isnull().mean() * 100
    for col, pct in missing_pct.items():
        if pct > 0:
            if pct < 5:
                recommendations.append(f"Column '{col}' has {pct:.1f}% missing values. Consider dropping rows.")
            elif pct < 50:
                recommendations.append(f"Column '{col}' has {pct:.1f}% missing values. Consider imputing (mean/median/mode).")
            else:
                recommendations.append(f"Column '{col}' has {pct:.1f}% missing values. Consider dropping the column.")

    # 2. Skewness (Numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        skewness = df[numeric_cols].skew(numeric_only=True)
        for col, skew in skewness.items():
            if abs(skew) > 1.0:
                recommendations.append(f"Column '{col}' is highly skewed ({skew:.2f}). Consider log or box-cox transformation.")

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
                 recommendations.append(f"Column '{col}' has {outliers} outliers ({pct_outliers:.1f}%). Consider handling them.")

    # 4. High Correlation
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        for col in to_drop:
             recommendations.append(f"Column '{col}' is highly correlated with other features (>0.9). Consider removing it to reduce multicollinearity.")

    # 5. Categorical High Cardinality
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count > 50:
             recommendations.append(f"Column '{col}' has high cardinality ({unique_count} unique values). Consider frequency encoding or reduction.")
    
    # 6. Constant Columns
    for col in df.columns:
        if df[col].nunique() <= 1:
             recommendations.append(f"Column '{col}' has only 1 unique value. Consider dropping it.")

    return recommendations
