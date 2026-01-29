import pandas as pd
import numpy as np

def bin_features(df, columns, method, bins=5):
    """Bin continuous features into categorical"""
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
        
        try:
            if method == 'equal_width':
                df_copy[f'{col}_binned'] = pd.cut(df_copy[col], bins=bins, labels=False)
            elif method == 'equal_freq':
                df_copy[f'{col}_binned'] = pd.qcut(df_copy[col], q=bins, labels=False, duplicates='drop')
            elif method == 'custom':
                df_copy[f'{col}_binned'] = pd.cut(df_copy[col], bins=bins, labels=False)
        except Exception:
            pass
            
    return df_copy
