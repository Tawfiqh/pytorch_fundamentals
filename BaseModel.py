from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from torch.nn import functional as F


class BaseModel:
    def fit(self, dataset):
        raise Exception(
            "To be implemented -- this should be implemented in the subclass."
        )

    def predict(self, X):
        return self.forward(X)

    def score(self, X, y):
        r2_error = r2_score(
            self(X).detach().numpy(),
            y.detach().numpy(),
        )
        return r2_error

    def score_all(self, train, test, val):
        train_score = self.score(*train)
        test_score = self.score(*test)
        val_score = self.score(*val)
        return train_score, test_score, val_score

    def score_mse(self, X, y):
        mse = F.mse_loss(
            self(X).detach(),
            y.detach(),
        )
        return mse

    def score_all_mse(self, train, test, val):
        train_score = self.score_mse(*train)
        test_score = self.score_mse(*test)
        val_score = self.score_mse(*val)
        return train_score, test_score, val_score

    def mae(self, data_set):
        X_df = pd.DataFrame(data_set[0])
        y = pd.DataFrame(data_set[1])

        y_hat = self.predict(data_set[0])
        y_hat = pd.DataFrame(y_hat)

        X_df["y"] = y
        X_df["y_hat"] = y_hat

        # print(f"X_df[y] -  {X_df['y']}")
        # print(f"X_df[y_hat] {X_df['y_hat']}")

        X_df["error"] = X_df["y"] - X_df["y_hat"]
        X_df["absolute_error"] = X_df["error"].abs()
        mean_absolute_error = np.mean(X_df["absolute_error"])
        return mean_absolute_error
