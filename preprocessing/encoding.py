import pandas as pd
from sklearn.preprocessing import LabelEncoder

def one_hot_encode(df, columns):
    """Apply one-hot encoding to categorical columns"""
    if not columns:
        return df
    return pd.get_dummies(df, columns=columns, prefix=columns, drop_first=False)

def label_encode(df, columns):
    """Apply label encoding to categorical columns"""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    return df_copy

def target_encoding(df, categorical_cols, target_col, smoothing=1.0):
    """Target encoding for categorical variables"""
    df_copy = df.copy()
    
    if target_col not in df_copy.columns:
        return df_copy
    
    global_mean = df_copy[target_col].mean()
    
    for col in categorical_cols:
        if col in df_copy.columns:
            # Calculate target mean for each category
            target_means = df_copy.groupby(col)[target_col].agg(['mean', 'count'])
            
            # Apply smoothing
            smoothed_means = (target_means['mean'] * target_means['count'] + 
                            global_mean * smoothing) / (target_means['count'] + smoothing)
            
            # Create new encoded column
            df_copy[f'{col}_target_encoded'] = df_copy[col].map(smoothed_means).fillna(global_mean)
    
    return df_copy

def frequency_encoding(df, columns):
    """Frequency encoding for categorical variables"""
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            freq_map = df_copy[col].value_counts(normalize=True).to_dict()
            df_copy[f'{col}_freq_encoded'] = df_copy[col].map(freq_map)
    
    return df_copy
