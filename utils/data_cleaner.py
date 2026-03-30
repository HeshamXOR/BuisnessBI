import pandas as pd

def auto_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically cleans a dataset by:
    1. Removing duplicate rows.
    2. Removing completely empty rows/columns.
    3. Dropping columns with >50% missing values.
    4. Dropping rows with missing values in the remaining columns 
       to ensure clean and accurate visuals.
    """
    if df is None or df.empty:
        return df
        
    cleaned_df = df.copy()
    
    # 1. Drop duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # 2. Drop completely empty rows and columns
    cleaned_df = cleaned_df.dropna(how='all', axis=0)
    cleaned_df = cleaned_df.dropna(how='all', axis=1)
    
    if cleaned_df.empty:
        return cleaned_df
        
    # 3. Drop columns that are mostly empty (> 50% null)
    threshold = len(cleaned_df) * 0.5
    cleaned_df = cleaned_df.dropna(thresh=threshold, axis=1)
    
    # 4. Drop remaining nulls to avoid skewing visualizations
    cleaned_df = cleaned_df.dropna()
    
    return cleaned_df
