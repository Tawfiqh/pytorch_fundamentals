import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from BaseModel import BaseModel
from torch.utils.data import DataLoader
from data.boston_dataset_torch import BostonDatasetTorchy
import multiprocessing as mp


class NeuralNetworkRegressor(BaseModel, torch.nn.Module):
    def __init__(self, n_features, n_labels):
        super().__init__()
        # print(f"n_features:{n_features}")
        # print(f"n_labels:{n_labels}")
        self.n_labels = n_labels

        middle_layer_size = 2 ** 4
        middle_layer_2_size = 2 ** 5
        middle_layer_3_size = 2 ** 10

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, middle_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_layer_size, middle_layer_2_size),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_layer_2_size, middle_layer_3_size),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_layer_3_size, n_labels),
        )

    def forward(self, X):
        return self.layers(X)

    def fit(
        model, dataset, epochs=100, lr=0.1, print_losses=False, print_losses_graph=True
    ):
        optimiser = torch.optim.SGD(model.parameters(), lr)  # create optimiser
        losses = []

        # cores = mp.cpu_count()py

        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

        for epoch in range(epochs):
            for X, y in dataloader:
                optimiser.zero_grad()
                y_hat = model(X)
                loss = F.mse_loss(y_hat, y)

                if print_losses:
                    print(f"loss:{loss}")
                loss.backward()  # Upgrades the .grad -- of each of the parameters (based on backpopulating through the NN)
                optimiser.step()
                losses.append(loss.item())
        if print_losses_graph:
            plt.plot(losses)
            plt.show()


if __name__ == "__main__":

    def get_train_test_val(dataset):
        n = len(dataset)
        train_len = int(n * 0.8)
        val_len = int(n * 0.1)

        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [train_len, val_len, len(dataset) - train_len - val_len]
        )
        return train_set, val_set, test_set

    print()
    boston_dataset = BostonDatasetTorchy()
    # for a, b in [boston[1], boston[100], boston[500]]:
    #     print(f"a: {a}")
    #     print(f"b: {b}")

    train_dataset, val_dataset, test_dataset = get_train_test_val(boston_dataset)

    # X_boston, y_boston = datasets.load_boston(return_X_y=True)
    example_X, example_y = train_dataset[0]
    linear_regressor_boston = NeuralNetworkRegressor(len(example_X), len(example_y))
    linear_regressor_boston.fit(
        train_dataset,
        epochs=1000,
        lr=0.005,
        print_losses=False,
        print_losses_graph=False,
    )

    from sklearn.metrics import r2_score

    r2_error = r2_score(
        linear_regressor_boston(boston_dataset.X).detach().numpy(),
        boston_dataset.y.detach().numpy(),
    )
    print(f"R^2 error: {r2_error}")

    output_to_csv = False
    if output_to_csv:
        import pandas as pd
        from datetime import datetime

        X_df = pd.DataFrame(boston_dataset.X)

        y_hat = linear_regressor_boston(boston_dataset.X).detach().numpy()
        y_hat_2 = linear_regressor_boston.predict(boston_dataset.X).detach().numpy()

        X_df["y"] = pd.DataFrame(boston_dataset.y)
        X_df["y_hat"] = pd.DataFrame(y_hat)
        X_df["y_hat_2"] = pd.DataFrame(y_hat_2)
        current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
        filename = f"LogisticRegressor_breast_cancer_{current_time}.csv"
        X_df.to_csv(filename)
        print(f"Output results to:{filename}")
