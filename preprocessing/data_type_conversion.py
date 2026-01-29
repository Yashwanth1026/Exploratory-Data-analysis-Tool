import pandas as pd

def convert_data_types(df, columns, target_type):
    """Convert data types of specified columns"""
    df_copy = df.copy()
    converted_cols = []
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        try:
            if target_type == 'datetime':
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            elif target_type == 'numeric':
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            elif target_type == 'category':
                df_copy[col] = df_copy[col].astype('category')
            elif target_type == 'string':
                df_copy[col] = df_copy[col].astype(str)
            elif target_type == 'boolean':
                df_copy[col] = df_copy[col].astype(bool)
            converted_cols.append(col)
        except Exception:
            pass
    
    return df_copy, converted_cols

def detect_data_type_issues(df):
    """Detect potential data type issues"""
    issues = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if it could be numeric
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if numeric_count / len(df) > 0.8:
                issues.append(f"{col}: Likely numeric (stored as object)")
            
            # Check if it could be datetime
            try:
                datetime_count = pd.to_datetime(df[col], errors='coerce').notna().sum()
                if datetime_count / len(df) > 0.8:
                    issues.append(f"{col}: Likely datetime (stored as object)")
            except:
                pass
    
    return issues
