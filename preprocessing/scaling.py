import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standard_scale(df, columns):
    """Apply standard scaling to numerical columns"""
    df_copy = df.copy()
    if columns:
        scaler = StandardScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy

def min_max_scale(df, columns, feature_range=(0, 1)):
    """Apply min-max scaling to numerical columns"""
    df_copy = df.copy()
    if columns:
        scaler = MinMaxScaler(feature_range=feature_range)
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy
