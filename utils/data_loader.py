"""
Data Loader Module
==================
Functions for loading, validating, and managing CSV datasets.
"""

import os
import pandas as pd
from typing import Dict, Optional


def load_csv(filepath: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.
        parse_dates: Optional list of columns to parse as dates.

    Returns:
        pd.DataFrame with loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or has no valid columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=parse_dates)

    if df.empty:
        raise ValueError(f"Dataset is empty: {filepath}")

    return df


def load_all_datasets(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load all standard datasets from the data directory.

    Args:
        data_dir: Path to the directory containing CSV files.

    Returns:
        Dictionary mapping dataset names to DataFrames.
    """
    dataset_configs = {
        "sales": {
            "file": "sales_data.csv",
            "parse_dates": ["date"]
        },
        "marketing": {
            "file": "marketing_data.csv",
            "parse_dates": ["date"]
        },
        "customers": {
            "file": "customers_data.csv",
            "parse_dates": None
        },
        "github": {
            "file": "github_repos.csv",
            "parse_dates": ["last_updated"]
        }
    }

    datasets = {}
    for name, config in dataset_configs.items():
        filepath = os.path.join(data_dir, config["file"])
        try:
            datasets[name] = load_csv(filepath, parse_dates=config["parse_dates"])
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️ Warning: Could not load {name} dataset: {e}")

    return datasets


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains the required columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.

    Returns:
        True if all required columns are present.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about a DataFrame.

    Returns:
        Dictionary with shape, dtypes, missing values, and memory usage.
    """
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns)
    }
