import pandas as pd
import numpy as np

def filter_dataframe(df, column, value, operator):
    """
    Filters the dataframe based on the column, value and operator.
    """
    if column not in df.columns:
        return df
    
    col_dtype = df[column].dtype
    
    # Ensure value is compatible if possible, though strict typing should be handled by caller
    
    if pd.api.types.is_numeric_dtype(col_dtype) or pd.api.types.is_datetime64_any_dtype(col_dtype):
        if operator == '>':
            return df[df[column] > value]
        elif operator == '<':
            return df[df[column] < value]
        elif operator == '==':
            return df[df[column] == value]
        elif operator == '!=':
            return df[df[column] != value]
        elif operator == '>=':
            return df[df[column] >= value]
        elif operator == '<=':
            return df[df[column] <= value]
    else:
        # String/Object operations
        str_val = str(value)
        if operator == '==':
            return df[df[column] == value]
        elif operator == '!=':
            return df[df[column] != value]
        elif operator == 'contains':
            return df[df[column].astype(str).str.contains(str_val, na=False)]
        elif operator == 'starts with':
            return df[df[column].astype(str).str.startswith(str_val, na=False)]
        elif operator == 'ends with':
            return df[df[column].astype(str).str.endswith(str_val, na=False)]
            
    return df

def drop_rows_based_on_filter(df, column, value, operator):
    """
    Drops rows based on the filter condition (inverse of filter).
    """
    if column not in df.columns:
        return df
        
    col_dtype = df[column].dtype
    
    if pd.api.types.is_numeric_dtype(col_dtype) or pd.api.types.is_datetime64_any_dtype(col_dtype):
        if operator == '>':
            return df[~(df[column] > value)]
        elif operator == '<':
            return df[~(df[column] < value)]
        elif operator == '==':
            return df[~(df[column] == value)]
        elif operator == '!=':
            return df[~(df[column] != value)]
        elif operator == '>=':
            return df[~(df[column] >= value)]
        elif operator == '<=':
            return df[~(df[column] <= value)]
    else:
        str_val = str(value)
        if operator == '==':
            return df[~(df[column] == value)]
        elif operator == '!=':
            return df[~(df[column] != value)]
        elif operator == 'contains':
            return df[~(df[column].astype(str).str.contains(str_val, na=False))]
        elif operator == 'starts with':
            return df[~(df[column].astype(str).str.startswith(str_val, na=False))]
        elif operator == 'ends with':
            return df[~(df[column].astype(str).str.endswith(str_val, na=False))]
            
    return df

def search_rows_global(df, query, case=False):
    """
    Searches for the query string in all columns of the dataframe.
    Returns rows where any column contains the query.
    """
    if not query:
        return df
        
    mask = pd.Series(False, index=df.index)
    str_query = str(query)
    
    for col in df.columns:
        # Convert to string and check for containment
        try:
            col_mask = df[col].astype(str).str.contains(str_query, case=case, na=False, regex=False)
            mask |= col_mask
        except Exception:
            continue
            
    return df[mask]
