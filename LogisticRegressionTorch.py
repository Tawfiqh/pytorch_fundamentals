import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from os import path as os_path

from sklearn import datasets

# from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import f1_score


class LogisticRegressionTorchy(torch.nn.Module):
    def __init__(self, n_features, n_labels):
        super().__init__()
        print(f"n_features:{n_features}")
        print(f"n_labels:{n_labels}")
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_labels), torch.nn.Sigmoid()
        )

    def forward(self, X):
        return self.layers(X)

    def predict(self, X):
        result = self.forward(X)
        rounded_result = result > 0.5
        return rounded_result

    def fit(model, X, y, epochs=100, lr=0.1, print_losses=False):
        for param in model.parameters():
            print(f"Parameters are: {param}")
        optimiser = torch.optim.SGD(model.parameters(), lr)  # create optimiser
        losses = []
        for epoch in range(epochs):
            optimiser.zero_grad()
            y_hat = model(X)
            loss = F.binary_cross_entropy(y_hat.reshape(-1), y.reshape(-1))

            if print_losses:
                print(f"loss:{loss}")
            loss.backward()  # Upgrades the .grad -- of each of the parameters (based on backpopulating through the NN)
            optimiser.step()
            losses.append(loss.item())
        plt.plot(losses)
        plt.show()


X, y = datasets.load_breast_cancer(return_X_y=True)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25) # 0.25 x 0.8 = 0.2

# Normalize the data
sc = preprocessing.StandardScaler()
sc.fit(X)
X = sc.transform(X)


X = torch.Tensor(X)
y = torch.Tensor(y)


logistic_regressor = LogisticRegressionTorchy(X.shape[1], 1)
logistic_regressor.fit(X, y, epochs=500, lr=0.01, print_losses=False)

from sklearn.metrics import r2_score

f1_score = f1_score(logistic_regressor.predict(X).detach().numpy(), y.detach().numpy())
print(f"F1 error: {f1_score}")

output_to_csv = True
if output_to_csv:
    X_df = pd.DataFrame(X)

    y_hat = logistic_regressor(X).detach().numpy()
    y_hat_2 = logistic_regressor.predict(X).detach().numpy()

    X_df["y"] = pd.DataFrame(y)
    X_df["y_hat"] = pd.DataFrame(y_hat)
    X_df["y_hat_2"] = pd.DataFrame(y_hat_2)
    current_time = datetime.now().strftime("%Y_%b_%d-%H_%M")
    filename = f"LogisticRegressor_breast_cancer_{current_time}.csv"
    X_df.to_csv(filename)
    print(f"Output results to:{filename}")
