# train_deep_mlp.py

import torch
import torch.nn as nn
import torch.optim as optim
from dataset_utils import get_dataloader
from regularizers import l1_regularization, l2_regularization, adaptive_regularization
import matplotlib.pyplot as plt


class DeepMLP(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.activations = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x); self.activations.append(x.detach()); x = self.relu(x)
        x = self.fc2(x); self.activations.append(x.detach()); x = self.relu(x)
        x = self.fc3(x); self.activations.append(x.detach()); x = self.relu(x)
        x = self.fc4(x)
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
    input_size = 784 if args.dataset != "CIFAR10" else 1024
    model = DeepMLP(input_size=input_size).to(device)
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
    plt.title(f"Deep MLP - {args.dataset} - {args.reg.upper()} Regularization")
    plt.xlabel("Epoch"); plt.legend(); plt.grid()
    plt.savefig(f"deep_mlp_{args.dataset.lower()}_{args.reg.lower()}_plot.png")
    plt.show()
