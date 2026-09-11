from dataset import RegressionDataset
import torch
from plotter import TrainingPlotter
from torch.utils.data import DataLoader

data = RegressionDataset(20, 0.1, 0.01)
train_dataloader = DataLoader(data, batch_size=5, shuffle=True)
test_dataloader = DataLoader(data, batch_size=5, shuffle=True)

torch.manual_seed(67)

x_dim = 1
n = 10
epochs = 100000
alpha = 0.05
beta = 0.5

w = torch.rand((x_dim + 1, 1))


def f(X):
    return X @ w

def grad_loss(X, y):
    return X.T @ (X @ w - y)

def loss(X, y):
    return 0.5 * torch.linalg.vector_norm(X @ w - y, 2) ** 2


X, y = next(iter(train_dataloader))

plotter = TrainingPlotter(data.X, data.y, w, 1)

g = grad_loss(X, y)

epoch = 0
while epoch < epochs:
    for X, y in train_dataloader:
        epoch += 1

        grad = grad_loss(X, y)

        l = loss(X, y)

        print(f"Loss: {l}, epoch: {epoch}")

        plotter.update(
            w=w,
            loss=l,
            epoch=epoch,
            X_batch=X,
            y_batch=y
        )

        g = beta * g + (1 - beta) * grad

        w -= alpha * g


plotter.show()