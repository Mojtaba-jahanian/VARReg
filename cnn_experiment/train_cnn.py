# train_cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset_utils import get_dataloader
from regularizers import l1_regularization, l2_regularization, adaptive_regularization


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.activations = []

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        self.activations.append(x.detach())
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, dataloader, criterion, optimizer, reg_type="none", alpha=1e-4):
    model.train()
    losses, accuracies = [], []

    for epoch in range(10):
        total_loss, correct, total = 0, 0, 0
        model.activations = []

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            if reg_type == "l1":
                loss += alpha * l1_regularization(model)
            elif reg_type == "l2":
                loss += alpha * l2_regularization(model)
            elif reg_type == "adaptive":
                loss += adaptive_regularization(model, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            total_loss += loss.item() * y.size(0)
            model.activations = []

        losses.append(total_loss / total)
        accuracies.append(100.0 * correct / total)

    return losses, accuracies


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg", type=str, default="none", choices=["none", "l1", "l2", "adaptive"])
    parser.add_argument("--dataset", type=str, default="MNIST")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(args.dataset)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    if args.reg == "l2":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        reg_type = "none"
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        reg_type = args.reg

    loss, acc = train(model, dataloader, criterion, optimizer, reg_type)

    # Plot results
    plt.figure()
    plt.plot(loss, label="Loss")
    plt.plot(acc, label="Accuracy")
    plt.title(f"CNN - {args.dataset} - {args.reg.upper()} Regularization")
    plt.xlabel("Epoch"); plt.legend(); plt.grid()
    plt.savefig(f"cnn_{args.dataset.lower()}_{args.reg.lower()}_plot.png")
    plt.show()
