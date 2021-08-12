from data.get_data import (
    get_boston_train_test_val_datasets,
    get_diabetes_train_test_val_datasets,
    get_school_data_train_test_val_datasets,
    get_car_data_train_test_val_datasets,
)

from data.get_pytorch_data import get_boston_datasets

import pandas as pd
from time import perf_counter
from datetime import datetime
from NeuralNetworkClassifier import NeuralNetworkClassifier
from LinearRegressionTorch import LinearRegressorTorchy
from LogisticRegressionTorch import LogisticRegressionTorchy

import matplotlib.pyplot as plt

from os import path as os_path


def get_target_shape(y):

    if len(y.shape) > 1:
        return y.shape[1]
    else:  # If it's just a vector
        return 1


def run_all_models_on_dataset(
    models, data_set, dataset_name, output_to_csv=False, fit_hyper_parameters=False
):
    all_model_results = []

    for model_name in models.keys():
        print(f"     {model_name}")
        model_class = models[model_name]
        time_start = perf_counter()

        # Tune (if the model has a function for tuning)
        target_shape = get_target_shape(data_set["train"][1])
        # print(f"target_shape: {target_shape}")
        model = model_class(data_set["train"][0].shape[1], target_shape)

        # Tune + FIT
        model.fit(*data_set["train"])

        time_finished_fit = perf_counter()
        fit_time = time_finished_fit - time_start

        # SCORE ALL
        model_results = model.score_all(
            data_set["train"], data_set["test"], data_set["val"]
        )
        time_finished_scoring = perf_counter()
        scoring_time = time_finished_scoring - time_finished_fit

        model_mse_results = model.score_all_mse(
            data_set["train"], data_set["test"], data_set["val"]
        )

        model_mean_absolute_error = model.mae(data_set["whole"])
        # output the result
        if output_to_csv:
            X_df = pd.DataFrame(data_set["whole"][0])
            y = pd.DataFrame(data_set["whole"][1])

            y_hat = model.predict(data_set["whole"][0])
            y_hat = pd.DataFrame(y_hat)

            X_df["y"] = y
            X_df["y_hat"] = y_hat
            current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
            X_df.to_csv(
                os_path.join(
                    "CSV_outputs", f"{model_name}_{dataset_name}_{current_time}.csv"
                )
            )

        # print(f"model_results:{model_results}")
        if model_results:
            all_model_results.append(
                [
                    model_name,
                    fit_time,
                    scoring_time,
                    model_mean_absolute_error,
                    *model_results,
                    *model_mse_results,
                ]
            )

    df = pd.DataFrame(
        all_model_results,
        columns=[
            "model_name",
            "fit_time",
            "scoring_time",
            "mean_absolute_error",
            "training_r^2_score",
            "testing_r^2_score",
            "validation_r^2_score",
            "training_mse_score",
            "testing_mse_score",
            "validation_mse_score",
        ],
    )
    pd.options.display.float_format = "{:,.4f}".format

    if output_to_csv:
        current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
        df.to_csv(os_path.join("CSV_outputs", f"results_df_{current_time}.csv"))

    print(df)
    print()

    best_result = df[df["validation_r^2_score"] == df["validation_r^2_score"].max()]
    print("Best model result:")
    print(best_result["model_name"].head(1).item())


boston_data_set = get_boston_train_test_val_datasets()
diabetes_data_set = get_diabetes_train_test_val_datasets()
school_results_data_set = get_school_data_train_test_val_datasets()
car_results_data_set = get_car_data_train_test_val_datasets()

datasets = [
    ("boston_data_set", boston_data_set),
    ("diabetes_data_set", diabetes_data_set),
    ("school_results_data_set", school_results_data_set),
    ("car_results_data_set", car_results_data_set),
]


for data_set_name, data_set in datasets:
    models = {
        "LinearRegressorTorchy": LinearRegressorTorchy,
        "NeuralNetworkClassifier": NeuralNetworkClassifier,
    }

    print()
    print(f"EVALUATING {data_set_name}")
    run_all_models_on_dataset(models, data_set, data_set_name, output_to_csv=True)


# %%
