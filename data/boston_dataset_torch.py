import torch
from sklearn import datasets
from sklearn import preprocessing


class BostonDatasetTorchy(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        X, y = datasets.load_boston(return_X_y=True)

        # Normalize the data
        sc = preprocessing.StandardScaler()
        sc.fit(X)
        X = sc.transform(X)

        X = torch.Tensor(X)
        y = torch.Tensor(y)
        y = y.reshape(-1, 1)

        self.X = X
        self.y = y

    # Not dependent on index
    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        length = self.X.shape[1]
        # len(self.y)
        return length
