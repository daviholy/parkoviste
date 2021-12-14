from torch import nn
from torch.optim import Adam
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, classes, device="cpu"):
        super().__init__()
        self.device = device
        self.classes = classes

        self.layer_input = nn.Sequential(nn.AdaptiveMaxPool2d(120))
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 1), nn.Conv2d(
            32, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.Conv2d(
            64, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.Conv2d(
            128, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(3))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Flatten())
        self.layer_output = nn.Sequential(
            nn.Identity(), nn.BatchNorm1d(2304), nn.ReLU(), nn.Identity(), nn.BatchNorm1d(2304), nn.ReLU(),  nn.LazyLinear(1), nn.Sigmoid())

    def forward(self, x):
        x = self.layer_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer_output(x)

    def train_model(self, data_loader, num_epochs, learning_rate):

        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)

        n_total_steps = len(data_loader)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(data_loader):

                images, labels = images.to(self.device), labels.to(self.device)

                output = self(images)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 2000 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

        print("Finished Training")

    def evaluate_model(self, data_loader):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0, 0]
            n_class_samples = [0, 0]

            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)

                print(outputs)

                predicted = torch.round(outputs)
                print(predicted)
                print(labels)

                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                # print(n_correct)
                # print("----------------")

                for i in range(len(labels)):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[int(label.item())] += 1
                    n_class_samples[int(label.item())] += 1

            acc = 100.0 * n_correct / n_samples
            # print(n_samples)
            print(f"Accuracy of the net: {acc}%")

            for i in range(len(self.classes)):
                if n_class_samples[i] == 0:
                    n_class_samples[i] = 1
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f"Accuracy of {self.classes[i]}: {acc}%")
