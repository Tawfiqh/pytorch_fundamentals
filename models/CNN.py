import torch
import torchvision


from torch.utils.tensorboard import SummaryWriter


class ConvNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 10),  # 10 output classes
            torch.nn.ReLU(),
            torch.nn.Linear(10, 128),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 4 * 4)  # Turn it into a vector as opposed to a 2d-image
        x = self.linear_layers(x)

        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.softmax(x)
        return x


if __name__ == "__main__":

    def train(model, loader, epochs=10):
        writer = SummaryWriter()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        batch_idx = 0
        for epoch in range(epochs):
            for features, labels in iter(loader):
                # print(labels)
                optimizer.zero_grad()
                output = model(features)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                writer.add_scalar("loss/train", loss.item(), batch_idx)
                batch_idx += 1
            print(f"{epoch} / {epochs}: -- Loss= {loss.item()}")

    mnist = torchvision.datasets.MNIST(
        root="./",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=64, shuffle=True)

    print(f"mnist-data: {mnist}")
    # print(f"mnist-data: {mnist[0]}")
    # print(mnist[0][0].show())

    cnn = ConvNet()
    train(cnn, loader)
