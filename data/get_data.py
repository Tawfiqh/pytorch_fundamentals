from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from .car_data import load_cleaned_car_data
from .school_data import load_cleaned_school_data
import torch


def generate_random_seed():
    return 3
    # Keep this fixed for now -- so that every time we call this (for different models) it returns the same split
    # Want to keep it fixed so we can compare across different runs.
    # Can change this function to be more dyanmic in future.


random_seed = generate_random_seed()


def torch_version(dataset):
    X = torch.Tensor(dataset[0])
    y = torch.Tensor(list(dataset[1]))
    y = y.reshape(-1, 1)  # 1 as we know it only has one target variable.

    return (X, y)


def split_into_train_test_val_datasets(X, y, normalize=True):

    # random-state fr test_split will default to using the global random state instance from numpy.random. Calling the function multiple times will reuse the same instance, and will produce different results.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2  # , random_state=random_seed
    )
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train, y_train, test_size=0.25  # , random_state=random_seed
    )  # 0.25 x 0.8 = 0.2

    # Normalise data before returning it
    if normalize:
        sc = preprocessing.StandardScaler()
        sc.fit(X_train)
        X_train_normalised = sc.transform(X_train)
        X_test_normalised = sc.transform(X_test)
        X_val_normalised = sc.transform(X_val)
        X_whole_normalised = sc.transform(X)

        train = (X_train_normalised, y_train)
        test = (X_test_normalised, y_test)
        val = (X_val_normalised, y_val)
        whole_dataset = (X_whole_normalised, y)
    else:
        train = (X_train, y_train)
        test = (X_test, y_test)
        val = (X_val, y_val)
        whole_dataset = (X, y)

    train = torch_version(train)
    test = torch_version(test)
    val = torch_version(val)
    whole_dataset = torch_version(whole_dataset)

    return {"train": train, "test": test, "val": val, "whole": whole_dataset}


def get_boston_train_test_val_datasets():
    X, y = datasets.load_boston(return_X_y=True)
    return split_into_train_test_val_datasets(X, y)


def get_diabetes_train_test_val_datasets():
    X, y = datasets.load_diabetes(return_X_y=True)
    return split_into_train_test_val_datasets(X, y)


def get_school_data_train_test_val_datasets():
    X, y = load_cleaned_school_data()
    return split_into_train_test_val_datasets(X, y)


def get_car_data_train_test_val_datasets():
    X, y = load_cleaned_car_data()
    return split_into_train_test_val_datasets(X, y, normalize=True)
