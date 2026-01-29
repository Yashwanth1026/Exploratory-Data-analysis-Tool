import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression

def feature_selection(df, method, target_col=None, threshold=0.8, k=10):
    """Advanced feature selection methods"""
    df_copy = df.copy()
    
    if method == 'variance':
        # Remove low variance features
        selector = VarianceThreshold(threshold=threshold)
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_data = selector.fit_transform(df_copy[numeric_cols])
            selected_cols = numeric_cols[selector.get_support()]
            
            # Keep non-numeric columns
            non_numeric_cols = df_copy.select_dtypes(exclude=[np.number]).columns
            result_df = pd.concat([
                pd.DataFrame(selected_data, columns=selected_cols, index=df_copy.index),
                df_copy[non_numeric_cols]
            ], axis=1)
            
            return result_df, len(numeric_cols) - len(selected_cols)
    
    elif method == 'correlation':
        # Remove highly correlated features
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df_copy[numeric_cols].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]
            
            result_df = df_copy.drop(columns=to_drop)
            return result_df, len(to_drop)
    
    elif method == 'univariate' and target_col and target_col in df_copy.columns:
        # Univariate feature selection
        X = df_copy.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = df_copy[target_col]
        
        if len(X.columns) > 0:
            # Determine if regression or classification
            if pd.api.types.is_numeric_dtype(y):
                selector = SelectKBest(f_regression, k=min(k, len(X.columns)))
            else:
                selector = SelectKBest(f_classif, k=min(k, len(X.columns)))
            
            X_selected = selector.fit_transform(X, y)
            selected_cols = X.columns[selector.get_support()]
            
            # Combine with non-numeric columns and target
            non_numeric_cols = df_copy.select_dtypes(exclude=[np.number]).columns
            result_df = pd.concat([
                pd.DataFrame(X_selected, columns=selected_cols, index=df_copy.index),
                df_copy[non_numeric_cols],
                df_copy[[target_col]]
            ], axis=1)
            
            return result_df, len(selected_cols)
    
    return df_copy, 0
