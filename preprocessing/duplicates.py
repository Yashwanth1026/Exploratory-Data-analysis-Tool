import pandas as pd

def detect_duplicates(df):
    """Detect duplicate rows"""
    duplicates = df.duplicated()
    duplicate_count = duplicates.sum()
    
    return {
        'total_duplicates': duplicate_count,
        'duplicate_percentage': (duplicate_count / len(df)) * 100,
        'duplicate_indices': df[duplicates].index.tolist()
    }

def remove_duplicates(df, subset=None, keep='first'):
    """Remove duplicate rows"""
    original_length = len(df)
    df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = original_length - len(df_cleaned)
    
    return df_cleaned, removed_count
