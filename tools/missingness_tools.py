import pandas as pd


def analyze_numeric_column(df, column):

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if series.empty:
        raise ValueError(
            f"Column '{column}' contains no numeric values."
        )

    return {
        "column": column,
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "skewness": float(series.skew()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
    }

def analyze_categorical_column(df, column):

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    series = df[column].dropna()

    if series.empty:
        raise ValueError(
            f"Column '{column}' contains no values."
        )

    value_counts = (
        series
        .value_counts()
        .head(10)
        .to_dict()
    )

    return {
        "column": column,
        "unique_values": int(series.nunique()),
        "top_categories": value_counts,
        "most_frequent": (
            series.mode().iloc[0]
            if not series.mode().empty
            else None
        )
    }


def analyze_missingness_by_category(
    df,
    target_column,
    grouping_column
):

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if grouping_column not in df.columns:
        raise ValueError(
            f"Grouping column '{grouping_column}' does not exist."
        )

    missing_indicator = df[target_column].isna()

    result = (
        df.assign(
            _missing=missing_indicator
        )
        .groupby(grouping_column)["_missing"]
        .agg(
            missing_count="sum",
            total_count="count"
        )
    )

    result["missing_percentage"] = (
        result["missing_count"]
        / result["total_count"]
        * 100
    )

    return result.reset_index().to_dict(
        orient="records"
    )

if __name__ == "__main__":

    test_df = pd.DataFrame({
        "age": [20, 21, None, 24, 25],
        "income": [30000, 35000, None, 90000, 100000],
        "gender": ["M", "F", "F", "M", "F"]
    })

    print("\nNUMERIC ANALYSIS")
    print(
        analyze_numeric_column(
            test_df,
            "income"
        )
    )

    print("\nCATEGORICAL ANALYSIS")
    print(
        analyze_categorical_column(
            test_df,
            "gender"
        )
    )

    print("\nMISSINGNESS BY CATEGORY")
    print(
        analyze_missingness_by_category(
            test_df,
            "age",
            "gender"
        )
    )