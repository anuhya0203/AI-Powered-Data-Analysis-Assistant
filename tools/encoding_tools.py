import pandas as pd


def is_categorical_column(series):
    """
    Determine whether a pandas Series should be treated
    as a categorical feature.
    """

    # Object columns
    if pd.api.types.is_object_dtype(series):
        return True

    # StringDtype columns
    if pd.api.types.is_string_dtype(series):
        return True

    # Pandas categorical dtype
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True

    # Boolean columns
    if pd.api.types.is_bool_dtype(series):
        return True

    return False


def analyze_categorical_column(df, column):

    series = df[column]

    non_missing = series.dropna()

    total_count = len(series)

    unique_count = int(
        series.nunique(dropna=True)
    )

    unique_percentage = (
        unique_count / len(non_missing) * 100
        if len(non_missing) > 0
        else 0
    )

    # Category frequencies
    value_counts = (
        non_missing
        .value_counts()
    )

    top_categories = (
        value_counts
        .head(10)
        .to_dict()
    )

    # Most frequent category percentage
    most_frequent_percentage = (
        value_counts.iloc[0] / len(non_missing) * 100
        if len(value_counts) > 0
        else 0
    )

    # Average frequency per category
    average_frequency = (
        len(non_missing) / unique_count
        if unique_count > 0
        else 0
    )

    # Cardinality classification
    low_cardinality = (
        unique_count <= 10
    )

    high_cardinality = (
        unique_count > 20
        or unique_percentage > 10
    )

    # Sample categories
    sample_categories = (
        non_missing
        .astype(str)
        .drop_duplicates()
        .head(20)
        .tolist()
    )

    return {

        "column": column,

        "dtype": str(
            series.dtype
        ),

        "total_count": total_count,

        "missing_count": int(
            series.isnull().sum()
        ),

        "unique_count": unique_count,

        "unique_percentage": round(
            unique_percentage,
            2
        ),

        "low_cardinality": low_cardinality,

        "high_cardinality": high_cardinality,

        "most_frequent_percentage": round(
            most_frequent_percentage,
            2
        ),

        "average_frequency": round(
            average_frequency,
            2
        ),

        "top_categories": top_categories,

        "sample_categories": sample_categories
    }


def analyze_categorical_features(
    df,
    target_column=None
):

    results = []

    for column in df.columns:

        # Never encode the target
        if column == target_column:
            continue

        if is_categorical_column(
            df[column]
        ):

            results.append(
                analyze_categorical_column(
                    df,
                    column
                )
            )

    return results