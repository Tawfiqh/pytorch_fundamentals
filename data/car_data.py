# %%
import pandas as pd
from os import path as os_path
from datetime import datetime


def load_dataframe():
    return pd.read_csv("car_data.csv")


def one_hot_encoding(df, columns):
    df = pd.get_dummies(df, columns=columns, drop_first=True,)
    return df


def frequency_encoding(df_column):
    counts = df_column.value_counts()

    col_length = len(df_column)

    df_column = df_column.apply(lambda classroom: counts[classroom] / col_length)
    # display(df_column)
    return df_column


def one_hot_encoding_from_list_column(df, column_name, enum_list):
    def contains_category(category, category_list):
        if category in category_list.split(","):
            return 1
        return 0

    df[column_name] = df[column_name].fillna("")

    for category in enum_list:
        df[f"{column_name}_{category}"] = df[column_name].apply(
            lambda category_list: contains_category(category, category_list)
        )
    return df


def mean_encoding(df, encoding_column_title, target_variable):
    # Assuming you want to replace the make with the average price of the make
    averages = {}

    def average_for_make(make):
        if make not in averages:
            average_for_make = df[df[encoding_column_title] == make][
                target_variable
            ].mean()

            averages[make] = average_for_make

        return averages[make]

    return df[encoding_column_title].apply(lambda value: average_for_make(value))


# Needs to return a tuple - (X, y)
def load_cleaned_car_data(return_df=False):
    df = load_dataframe()
    target_variable = "MSRP"

    df = one_hot_encoding(
        df,
        [
            "Engine Fuel Type",
            "Transmission Type",
            "Driven_Wheels",
            "Vehicle Size",
            "Vehicle Style",
        ],
    )

    df = one_hot_encoding_from_list_column(
        df,
        "Market Category",
        [
            "High-Performance",
            "Performance",
            "Hybrid",
            "Luxury",
            "Diesel",
            "Factory Tuner",
            "Flex Fuel",
            "Hatchback",
            "Exotic",
            "Crossover",
        ],
    )

    df["Make"] = mean_encoding(df, "Make", target_variable)
    df["Model"] = mean_encoding(df, "Model", target_variable)

    # Drop extra columns
    df = df.drop(["Market Category", "Popularity"], axis=1)

    # Remove NaNs
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    if return_df:
        print("outputting to clean CSV")
        current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
        df.to_csv(
            os_path.join("CSV_outputs", f"car_data_after_cleaning_{current_time}.csv")
        )
        return df

    y = df[target_variable]
    X = df.drop([target_variable], axis=1)
    return X, y


# Uncomment to run this as a cell and interact with the df variable
# df = load_cleaned_car_data(return_df=True)
# display(df)


# Extra data analytics we ran when inspecting the data
# %%
# display(df.isna().sum())  # Check for NaNs
# display(df.describe())
# display(df.dtypes)

# print("DUPLICATES:::")
# duplicated = df.duplicated(keep='last')
# display(df[duplicated])
