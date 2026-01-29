import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

def knn_imputation(df, columns, n_neighbors=5):
    """KNN imputation for missing values"""
    df_copy = df.copy()
    
    if not columns:
        return df_copy
    
    # Only apply to numeric columns
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df_copy[col])]
    
    if numeric_cols:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
    
    return df_copy

def interpolation_imputation(df, columns, method='linear'):
    """Interpolation imputation for time series data"""
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            if method == 'linear':
                df_copy[col] = df_copy[col].interpolate(method='linear')
            elif method == 'polynomial':
                df_copy[col] = df_copy[col].interpolate(method='polynomial', order=2)
            elif method == 'spline':
                df_copy[col] = df_copy[col].interpolate(method='spline', order=2)
    
    return df_copy

def regression_imputation(df, columns, target_col):
    """Use regression to predict missing values"""
    df_copy = df.copy()
    
    if target_col not in df_copy.columns:
        return df_copy
    
    # Prepare data
    X = df_copy.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    y = df_copy[target_col]
    
    # Find rows with missing target values
    missing_mask = y.isna()
    
    if missing_mask.sum() > 0 and not X.empty:
        # Train on complete cases
        X_train = X[~missing_mask].fillna(X.mean())
        y_train = y[~missing_mask]
        
        # Predict missing values
        X_predict = X[missing_mask].fillna(X.mean())
        
        if len(X_train) > 0 and len(X_predict) > 0:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_predict)
            df_copy.loc[missing_mask, target_col] = predictions
            
            return df_copy, missing_mask.sum()
            
    return df_copy, 0
