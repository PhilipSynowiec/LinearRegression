from sympy import plot

from dataset import RegressionDataset
import torch
from plotter import TrainingPlotter
from torch.utils.data import DataLoader

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

x_dim = 1
epochs = 100000
batch_size = 5
alpha = 0.05
beta = 0.5

data = RegressionDataset(100, 0.1, 0.01)
train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)


def model(X):
    return X @ w


def grad_loss(X, y):
    return X.T @ (X @ w - y)


def loss_fn(pred, y):
    v = pred - y
    return 0.5 * (v.T @ v)

def train_loop():
    global w
    grad = 0
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        grad = beta * grad + (1 - beta) * grad_loss(X, y)
        w -= alpha * grad

        if batch % 10 == 0:
            current = batch * batch_size + len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def test_loop():
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


w = torch.rand((x_dim + 1, 1)) *10
plotter = TrainingPlotter(data.X, data.y, w, pause=0.001)
for t in range(epochs):
    plotter.update(w=w)
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop()
    test_loop()