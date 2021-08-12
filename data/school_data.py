# %%
import pandas as pd


def load_dataframe():

    return pd.read_csv("test_scores.csv")


def one_hot_encoding(df, columns):
    df = pd.get_dummies(df, columns=columns, drop_first=True,)
    return df


def frequency_encoding(df_column):
    counts = df_column.value_counts()

    col_length = len(df_column)

    df_column = df_column.apply(
        lambda classroom: int((counts[classroom] / col_length) * 10000)
    )  # save space with integer div //
    # display(df_column)
    return df_column


# Needs to return a tuple - (X, y)
def load_cleaned_school_data(return_df=False):
    df = load_dataframe()
    df = one_hot_encoding(
        df,
        [
            "gender",
            "lunch",
            "teaching_method",
            "school_type",
            "school_setting",
            "school",  # there's about 20 of these, so we might try toggling this off later
        ],
    )
    df["classroom"] = frequency_encoding(df["classroom"])

    # todo - drop extra columns
    df = df.drop(["student_id"], axis=1)

    # display(df)
    if return_df:
        return df

    y = df["posttest"]
    X = df.drop(["posttest"], axis=1)
    return X, y


# df = load_cleaned_school_data(return_df=True)

# Extra data analytics we did when inspecting the data
# %%
# display(df.isna().sum()) # Check for NaNs
# display(df.describe())
# display(df.dtypes)
# # Replace float data-types with int
