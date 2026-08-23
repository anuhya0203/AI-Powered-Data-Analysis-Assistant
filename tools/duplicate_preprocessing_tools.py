import pandas as pd


def apply_duplicate_action(df, action):
    """
    Apply the selected duplicate-handling action.
    """

    if action == "Remove Duplicates":

        cleaned_df = df.drop_duplicates().copy()

        return cleaned_df

    elif action == "Keep Duplicates":

        return df.copy()

    else:

        raise ValueError(
            f"Unknown duplicate action: {action}"
        )