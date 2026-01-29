import pandas as pd
import numpy as np

def get_missing_values_summary(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    return missing_data

def remove_rows_with_missing_data(df, columns):
    """Remove rows with missing data in specified columns"""
    if columns:
        return df.dropna(subset=columns)
    return df

def fill_missing_data(df, columns, method):
    """Fill missing data using specified method"""
    df_copy = df.copy()
    for column in columns:
        if column in df_copy.columns:
            if method == 'mean' and pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column].fillna(df_copy[column].mean(), inplace=True)
            elif method == 'median' and pd.api.types.is_numeric_dtype(df_copy[column]):
                df_copy[column].fillna(df_copy[column].median(), inplace=True)
            elif method == 'mode':
                mode_val = df_copy[column].mode()
                if len(mode_val) > 0:
                    df_copy[column].fillna(mode_val.iloc[0], inplace=True)
    return df_copy
