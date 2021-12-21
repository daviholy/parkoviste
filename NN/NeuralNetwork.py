from torch import nn
from torch.optim import Adam
import torch
from torch.utils.data.dataloader import DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.layer_input = nn.Sequential(nn.AdaptiveMaxPool2d(120))
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 1), nn.Conv2d(32, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
                                    nn.MaxPool2d(3))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.MaxPool2d(3))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU(),
                                    nn.MaxPool2d(3))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Flatten())
        self.layer_output = nn.Sequential(
            nn.Identity(), nn.BatchNorm1d(2304), nn.ReLU(), nn.Identity(), nn.BatchNorm1d(2304), nn.ReLU(),  nn.LazyLinear(2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.layer_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer_output(x)

    def train_model(self, train_data_loader, test_data_loader, num_epochs, learning_rate):

        criterion = nn.NLLLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)

        n_total_steps = len(train_data_loader)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_data_loader):

                images, labels = images.to(self.device), labels.to(self.device)

                output = self(images)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 20 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")
            self.evaluate_model(test_data_loader, 0.4)

        print("Finished Training")

    def evaluate_model(self, data_loader: DataLoader, recall=0):
        if recall > 1 or recall < 0:
            raise Exception("the recall isn't in range <0,1>")
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = {}
            n_class_samples = {}

            for key in data_loader.dataset.classes.keys():  # initialize class value dict to 0
                n_class_correct[key] = 0
                n_class_samples[key] = 0

            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                predicted = self(images)

                # print(predicted)

                n_samples += len(labels)
                n_correct += sum([1 if _.max() <= recall else 0 for _ in labels - predicted])  # assuming the tensors with one 1

                for i in data_loader.dataset.classes.keys():
                    index = data_loader.dataset.classes[i].index(1)
                    idxs = [j for j in range(len(labels)) if labels[j][index] == 1.0]
                    n_class_correct[i] += sum([1 if recall >= (labels[idx][index] - predicted[idx][index]) else 0 for idx in idxs])
                    n_class_samples[i] += int(sum(labels[:, index]))

            acc = 100.0 * n_correct / n_samples
            print(f"Accuracy of the net: {acc}%")

            for i in data_loader.dataset.classes.keys():
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f"Accuracy of {data_loader.dataset.classes[i]}: {acc}%")

            print(f'total number of samples: {n_class_samples}')
            print(f'total number of correct guesses: {n_class_correct}')

