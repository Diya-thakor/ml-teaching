"""
Train a LeNet model on MNIST and save it for the Streamlit app.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)   # 28 -> 24
        self.conv2 = nn.Conv2d(6, 16, 5)  # 12 -> 8
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        self.features = x  # store feature maps
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_model():
    print("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    print("Initializing model...")
    model = LeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training for 5 epochs...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

    print("Saving model...")
    torch.save(model.state_dict(), 'lenet_mnist.pth')
    print("✓ Model saved as 'lenet_mnist.pth'")

    return model


if __name__ == "__main__":
    model = train_model()
