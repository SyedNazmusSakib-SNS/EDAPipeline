"""Utility functions for EDAPipeline."""

import numpy as np

def identify_column_types(df):
    """Helper function to identify column types in a dataframe."""
    
    
    numerical_cols = df.select_dtypes(include=np.number, exclude='bool').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Try to identify datetime columns
    datetime_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_datetime(df[col], errors='raise')
            datetime_cols.append(col)
        except:
            pass
            
    return numerical_cols, categorical_cols, datetime_cols