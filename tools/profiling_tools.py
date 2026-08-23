import pandas as pd


def profile_dataset(df):
    """
    Provides high-level information about the dataset.
    """

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(df.isnull().sum().sum()),
    }

    column_info = []

    for column in df.columns:

        series = df[column]

        column_info.append({
            "column": column,
            "dtype": str(series.dtype),
            "missing_count": int(series.isnull().sum()),
            "missing_percentage": round(
                series.isnull().mean() * 100,
                2
            ),
            "unique_values": int(
                series.nunique(dropna=True)
            )
        })

    profile["columns_info"] = column_info

    return profile