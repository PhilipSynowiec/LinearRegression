from dataset import RegressionDataset
import torch
from torch import nn
from plotter import TrainingPlotter
from torch.utils.data import DataLoader

x_dim = 1
epochs = 1000
batch_size = 5
alpha = 0.05
beta = 0.5

data = RegressionDataset(100, 0.1, 0.01)
train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NerualNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(nn.Linear(x_dim, 1))

    def forward(self, x):
        y = self.linear_stack(x)
        return y

model = NerualNetwork().to(device)
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum = beta)
loss_fn = nn.MSELoss()

def train_loop():
    size = len(train_dataloader.dataset)
    model.train() # here only for best practices
    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current = batch * batch_size + len(X)
        print(
            f"\rLoss: {loss.item():.6f}  "
            f"[{current:3d}/{size:3d}]",
            end="",
            flush=True
        )
    print()



def test_loop():
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")



plotter = TrainingPlotter(data.X, data.y, model, pause=0.001)
for t in range(epochs):
    plotter.update(model)
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop()
    test_loop()