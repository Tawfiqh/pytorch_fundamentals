import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

from BaseModel import BaseModel


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

    # def predict(self, X):
    #     result = self.forward(X)
    #     rounded_result = result > 0.5
    #     return rounded_result

    def fit(
        model, X, y, epochs=100, lr=0.1, print_losses=False, print_losses_graph=True
    ):
        # for param in model.parameters():
        #     print(f"Parameters are: {param}")
        optimiser = torch.optim.SGD(model.parameters(), lr)  # create optimiser
        losses = []
        for epoch in range(epochs):
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


# - name: My first pytorch dataset
#   description: |
#     - Load in the boston dataset ✅
#     - create a class called BostonDataset which inherits from torch.utils.data.Dataset ✅
#     - implement the two magic methods which need to be overwritten ✅
#       - refer to the torch docs if you cant remember which methods these are or why they are required ✅
#     - check that your dataset works by indexing it and asking for it’s len
#     - this doesn’t really look like we did anything specific to torch here. But the reason why we inherit from this class is because it acts as an interface, requiring us to implement things which are used by other torch methods, like the DataLoader
#     - remind yourself what an abstract class is.
#       - Discuss: Is torch.utils.data.Dataset an abstract class?
#         - look at the source code


class BostonDatasetTorchy(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        X, y = datasets.load_boston(return_X_y=True)
        self.X = X
        self.y = y

    # Not dependent on index
    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        length = self.X.shape[1]
        print(f"len: {length}")
        # len(self.y)
        return length


if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime

    from sklearn import datasets
    from sklearn import preprocessing

    X_boston, y_boston = datasets.load_boston(return_X_y=True)

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    # X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25) # 0.25 x 0.8 = 0.2

    # Normalize the data
    sc = preprocessing.StandardScaler()
    sc.fit(X_boston)
    X_boston = sc.transform(X_boston)

    X_boston = torch.Tensor(X_boston)
    y_boston = torch.Tensor(y_boston)
    y_boston = y_boston.reshape(-1, 1)

    linear_regressor_boston = NeuralNetworkRegressor(X_boston.shape[1], 1)
    linear_regressor_boston.fit(
        X_boston,
        y_boston,
        epochs=1000,
        lr=0.01,
        print_losses=False,
        print_losses_graph=False,
    )

    from sklearn.metrics import r2_score

    r2_error = r2_score(
        linear_regressor_boston(X_boston).detach().numpy(), y_boston.detach().numpy()
    )
    print(f"R^2 error: {r2_error}")

    output_to_csv = False
    if output_to_csv:
        X_df = pd.DataFrame(X_boston)

        y_hat = linear_regressor_boston(X_boston).detach().numpy()
        y_hat_2 = linear_regressor_boston.predict(X_boston).detach().numpy()

        X_df["y"] = pd.DataFrame(y_boston)
        X_df["y_hat"] = pd.DataFrame(y_hat)
        X_df["y_hat_2"] = pd.DataFrame(y_hat_2)
        current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
        filename = f"LogisticRegressor_breast_cancer_{current_time}.csv"
        X_df.to_csv(filename)
        print(f"Output results to:{filename}")
