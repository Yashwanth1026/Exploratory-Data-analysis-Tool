import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def apply_pca(df, n_components=None, variance_threshold=0.95):
    """Apply PCA for dimensionality reduction"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df, None
    
    X = df[numeric_cols].fillna(0)  # Fill NaN values
    
    if n_components is None:
        # Find number of components for variance threshold
        pca_temp = PCA()
        pca_temp.fit(X)
        cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create new dataframe with PCA components
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    
    # Add non-numeric columns back
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        result_df = pd.concat([pca_df, df[non_numeric_cols]], axis=1)
    else:
        result_df = pca_df
    
    pca_info = {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'n_components': n_components,
        'original_feature_count': len(numeric_cols)
    }
    
    return result_df, pca_info
