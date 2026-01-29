import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

def transform_features(df, columns, method):
    """Transform features using various methods"""
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
        
        if method == 'log':
            # Add small constant to handle zeros
            df_copy[f'{col}_log'] = np.log1p(df_copy[col].clip(lower=0))
        
        elif method == 'sqrt':
            df_copy[f'{col}_sqrt'] = np.sqrt(df_copy[col].clip(lower=0))
        
        elif method == 'square':
            df_copy[f'{col}_square'] = df_copy[col] ** 2
        
        elif method == 'reciprocal':
            # Avoid division by zero
            df_copy[f'{col}_reciprocal'] = 1 / (df_copy[col] + 1e-8)
        
        elif method == 'boxcox':
            # Box-Cox requires positive values
            if (df_copy[col] > 0).all():
                pt = PowerTransformer(method='box-cox')
                df_copy[f'{col}_boxcox'] = pt.fit_transform(df_copy[[col]]).flatten()
            else:
                pass 
        
        elif method == 'yeojohnson':
            pt = PowerTransformer(method='yeo-johnson')
            df_copy[f'{col}_yeojohnson'] = pt.fit_transform(df_copy[[col]]).flatten()
    
    return df_copy
