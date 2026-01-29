import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(df, column_name):
    """Detect outliers using IQR method"""
    if column_name not in df.columns:
        return []
    data = df[column_name].dropna()
    if len(data) == 0:
        return []
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    outliers = data[(data < lower) | (data > upper)].tolist()
    return sorted(outliers)

def detect_outliers_zscore(df, column_name, threshold=3):
    """Detect outliers using Z-score method"""
    if column_name not in df.columns:
        return []
    data = df[column_name].dropna()
    if len(data) == 0:
        return []
    z_scores = np.abs(stats.zscore(data))
    return data[z_scores > threshold].tolist()

def remove_outliers(df, column_name, outliers):
    """Remove outliers from dataframe"""
    if not outliers or column_name not in df.columns:
        return df
    return df[~df[column_name].isin(outliers)]

def transform_outliers(df, column_name, outliers, method='median'):
    """Transform outliers using specified method"""
    df_copy = df.copy()
    if not outliers or column_name not in df_copy.columns:
        return df_copy
    
    if method == 'median':
        replacement_value = df_copy[~df_copy[column_name].isin(outliers)][column_name].median()
    elif method == 'mean':
        replacement_value = df_copy[~df_copy[column_name].isin(outliers)][column_name].mean()
    else:  # cap
        q25, q75 = np.percentile(df_copy[column_name].dropna(), [25, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        df_copy[column_name] = df_copy[column_name].clip(lower=lower, upper=upper)
        return df_copy
    
    df_copy.loc[df_copy[column_name].isin(outliers), column_name] = replacement_value
    return df_copy
