import torch

from plotter import TrainingPlotter

torch.manual_seed(42)

x_dim = 1
n = 10
iters = 100000
alpha = 0.1
beta = 0.5

X = torch.rand((n, x_dim + 1))
X[:, -1] = 1

Y = X @ torch.rand((x_dim + 1, 1)) + (
    torch.randn((n, 1)) * 0.01 + 0.1
)

torch.rand((n, 1))

w = torch.rand((x_dim + 1, 1))


def f():
    return X @ w


def grad_L():
    return X.T @ (X @ w - Y)


def L():
    return 0.5 * torch.linalg.vector_norm(X @ w - Y, 2) ** 2


plotter = TrainingPlotter(X, Y, w, 1)

g = grad = grad_L()

for epoch in range(iters):

    loss = L()

    print(f"Loss: {loss}, epoch: {epoch}")

    plotter.update(
        w=w,
        loss=loss,
        epoch=epoch
    )

    g = beta * g + (1 - beta) * grad

    w -= alpha * g

    grad = grad_L()


plotter.show()