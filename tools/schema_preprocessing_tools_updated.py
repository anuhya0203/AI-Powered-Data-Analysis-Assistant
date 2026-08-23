import pandas as pd


def convert_to_numeric(df, column):
    result = df.copy()

    result[column] = pd.to_numeric(
        result[column],
        errors="coerce"
    )

    return result


def convert_to_datetime(df, column):
    result = df.copy()

    result[column] = pd.to_datetime(
        result[column],
        errors="coerce"
    )

    return result


def convert_to_categorical(df, column):
    result = df.copy()

    result[column] = (
        result[column].astype("category")
    )

    return result


def apply_schema_action(df, column, action):

    if action == "Convert to Numeric":

        return convert_to_numeric(
            df,
            column
        )

    elif action == "Convert to Datetime":

        return convert_to_datetime(
            df,
            column
        )

    elif action == "Convert to Categorical":

        return convert_to_categorical(
            df,
            column
        )

    elif action == "Drop Column":

        result = df.copy()

        return result.drop(
            columns=[column]
        )

    elif action == "Treat as Identifier":

        # Identifier columns are not modified.
        return df.copy()

    elif action == "No Action":

        return df.copy()

    else:

        raise ValueError(
            f"Unknown schema action: {action}"
        )


def apply_duplicate_action(df, action):

    if action == "Remove Duplicates":

        return df.drop_duplicates().copy()

    elif action == "Keep Duplicates":

        return df.copy()

    else:

        raise ValueError(
            f"Unknown duplicate action: {action}"
        )


def apply_schema_actions(
    df,
    selected_actions,
    duplicate_action="Keep Duplicates"
):
    """
    Apply duplicate handling first, followed by
    the selected schema actions.
    """

    cleaned_df = df.copy()
    summary = []

    duplicate_rows_before = int(
        cleaned_df.duplicated().sum()
    )

    rows_before_duplicates = len(cleaned_df)

    cleaned_df = apply_duplicate_action(
        cleaned_df,
        duplicate_action
    )

    rows_after_duplicates = len(cleaned_df)

    rows_removed_duplicates = (
        rows_before_duplicates
        - rows_after_duplicates
    )

    summary.append({
        "Column": "Dataset",
        "Action": duplicate_action,
        "Type Before": "N/A",
        "Type After": "N/A",
        "Rows": rows_before_duplicates,
        "Duplicate Rows": duplicate_rows_before,
        "Rows Removed": rows_removed_duplicates
    })

    for column, action in selected_actions.items():

        if column not in cleaned_df.columns:
            continue

        dtype_before = str(
            cleaned_df[column].dtype
        )

        rows_before = len(cleaned_df)

        cleaned_df = apply_schema_action(
            cleaned_df,
            column,
            action
        )

        if (
            action == "Drop Column"
            and column not in cleaned_df.columns
        ):

            dtype_after = "Dropped"

        else:

            dtype_after = str(
                cleaned_df[column].dtype
            )

        summary.append({
            "Column": column,
            "Action": action,
            "Type Before": dtype_before,
            "Type After": dtype_after,
            "Rows": rows_before
        })

    return cleaned_df, summary
