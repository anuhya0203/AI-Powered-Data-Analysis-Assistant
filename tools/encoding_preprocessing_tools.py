import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def apply_one_hot_encoding(
    df,
    column
):

    result = df.copy()

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded = encoder.fit_transform(
        result[[column]]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=[
            f"{column}_{category}"
            for category in encoder.categories_[0]
        ],
        index=result.index
    )

    result = result.drop(
        columns=[column]
    )

    result = pd.concat(
        [
            result,
            encoded_df
        ],
        axis=1
    )

    return result


def apply_frequency_encoding(
    df,
    column
):

    result = df.copy()

    frequencies = (
        result[column]
        .value_counts(
            normalize=True
        )
    )

    result[column] = (
        result[column]
        .map(frequencies)
    )

    return result


def apply_target_encoding(
    df,
    column,
    target_column
):

    result = df.copy()

    if target_column is None:
        raise ValueError(
            "Target column is required for "
            "Target Encoding."
        )

    if target_column not in result.columns:
        raise ValueError(
            "Target column does not exist."
        )

    mapping = (
        result.groupby(column)[
            target_column
        ]
        .mean()
    )

    result[column] = (
        result[column]
        .map(mapping)
    )

    return result


def apply_ordinal_encoding(
    df,
    column
):

    result = df.copy()

    categories = (
        result[column]
        .dropna()
        .unique()
        .tolist()
    )

    category_mapping = {
        category: index
        for index, category
        in enumerate(categories)
    }

    result[column] = (
        result[column]
        .map(category_mapping)
    )

    return result


def apply_encoding_action(
    df,
    column,
    action,
    target_column=None
):

    if action == "No Action":

        return df.copy()

    elif action == "One-Hot Encoding":

        return apply_one_hot_encoding(
            df,
            column
        )

    elif action == "Frequency Encoding":

        return apply_frequency_encoding(
            df,
            column
        )

    elif action == "Target Encoding":

        return apply_target_encoding(
            df,
            column,
            target_column
        )

    elif action == "Ordinal Encoding":

        return apply_ordinal_encoding(
            df,
            column
        )

    else:

        raise ValueError(
            f"Unknown encoding action: {action}"
        )


def apply_encoding_actions(
    df,
    selected_actions,
    target_column=None
):

    result = df.copy()

    summary = []

    for column, action in selected_actions.items():

        if column not in result.columns:
            continue

        if column == target_column:
            continue

        before_columns = len(
            result.columns
        )

        result = apply_encoding_action(
            result,
            column,
            action,
            target_column
        )

        after_columns = len(
            result.columns
        )

        summary.append({

            "Column": column,

            "Action": action,

            "Columns Before": before_columns,

            "Columns After": after_columns

        })

    return result, summary