import pandas as pd

def datetime_feature_engineering(df, datetime_col, features_to_extract=None):
    """Extract datetime features from datetime column"""
    df_copy = df.copy()
    
    if datetime_col not in df_copy.columns:
        return df_copy, []
    
    # Ensure column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy[datetime_col]):
        df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col], errors='coerce')
    
    if features_to_extract is None:
        features_to_extract = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'quarter', 'dayofyear']
    
    extracted_features = []
    
    for feature in features_to_extract:
        try:
            if feature == 'year':
                df_copy[f'{datetime_col}_year'] = df_copy[datetime_col].dt.year
                extracted_features.append('year')
            elif feature == 'month':
                df_copy[f'{datetime_col}_month'] = df_copy[datetime_col].dt.month
                extracted_features.append('month')
            elif feature == 'day':
                df_copy[f'{datetime_col}_day'] = df_copy[datetime_col].dt.day
                extracted_features.append('day')
            elif feature == 'weekday':
                df_copy[f'{datetime_col}_weekday'] = df_copy[datetime_col].dt.weekday
                extracted_features.append('weekday')
            elif feature == 'hour':
                df_copy[f'{datetime_col}_hour'] = df_copy[datetime_col].dt.hour
                extracted_features.append('hour')
            elif feature == 'minute':
                df_copy[f'{datetime_col}_minute'] = df_copy[datetime_col].dt.minute
                extracted_features.append('minute')
            elif feature == 'quarter':
                df_copy[f'{datetime_col}_quarter'] = df_copy[datetime_col].dt.quarter
                extracted_features.append('quarter')
            elif feature == 'dayofyear':
                df_copy[f'{datetime_col}_dayofyear'] = df_copy[datetime_col].dt.dayofyear
                extracted_features.append('dayofyear')
        except Exception:
            pass
            
    return df_copy, extracted_features

def create_date_differences(df, date_col1, date_col2, unit='days'):
    """Create date difference features"""
    df_copy = df.copy()
    
    if date_col1 not in df_copy.columns or date_col2 not in df_copy.columns:
        return df_copy
    
    # Ensure columns are datetime
    df_copy[date_col1] = pd.to_datetime(df_copy[date_col1], errors='coerce')
    df_copy[date_col2] = pd.to_datetime(df_copy[date_col2], errors='coerce')
    
    diff = df_copy[date_col2] - df_copy[date_col1]
    
    if unit == 'days':
        df_copy[f'{date_col2}_{date_col1}_days'] = diff.dt.days
    elif unit == 'hours':
        df_copy[f'{date_col2}_{date_col1}_hours'] = diff.dt.total_seconds() / 3600
    elif unit == 'minutes':
        df_copy[f'{date_col2}_{date_col1}_minutes'] = diff.dt.total_seconds() / 60
    
    return df_copy
