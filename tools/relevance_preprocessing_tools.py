import pandas as pd


def apply_relevance_actions(
    df,
    selected_actions
):

    result = df.copy()

    summary = []

    for column, action in selected_actions.items():

        if column not in result.columns:
            continue

        if action == "Drop":

            result = result.drop(
                columns=[column]
            )

            summary.append({

                "Column": column,

                "Action": "Dropped"

            })

        elif action == "Keep":

            summary.append({

                "Column": column,

                "Action": "Kept"

            })

        elif action == "Investigate":

            summary.append({

                "Column": column,

                "Action": "Investigate"

            })

    return result, summary