import pandas as pd

def remove_selected_columns(df, columns_remove):
    """Remove selected columns from dataframe"""
    if not columns_remove:
        return df
    return df.drop(columns=columns_remove, errors='ignore')

def categorical_numerical(df):
    """Function to find categorical and numerical columns/variables in dataset"""
    num_columns, cat_columns = [], []
    for col in df.columns:
        if len(df[col].unique()) <= 30 or df[col].dtype == object:
            cat_columns.append(col.strip())
        else:
            num_columns.append(col.strip())
    return num_columns, cat_columns
