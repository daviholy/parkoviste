import random
import numpy as np
import csv
from torch import nn, zeros, tensor, Tensor
from torch.optim import Adam
import torch
from torch.utils.data.dataloader import DataLoader
from NN.ModelStatistics import *
from math import floor, ceil
from torch.nn import ConstantPad2d


class MyAdaptiveAvgPool2d():
    def __init__(self, x, y):
        super().__init__()
        self.y = y
        self.x = x

    def kernel(self, x_out, x_in):
        if x_out == x_in:
            return (tensor(1), tensor(1), 0, 0)
        # calculate how many times we need to repeat the bigger kernel
        f = floor(x_in / x_out)
        c = ceil(x_in / x_out)
        n = ((f * x_out) - x_in) / (f - c)
        if n % 2 == 0:
            return c, f, int(n / 2), int(n / 2)
        else:
            return c, f, int((n + 1) / 2), int((n - 1) / 2)

    def forward(self, x) -> Tensor:

        # maxpooling the bigger dimensions
        result = zeros(x.shape[0], x.shape[1], self.y, self.x)
        kernel_x = self.kernel(self.x, x.shape[3])
        kernel_y = self.kernel(self.y, x.shape[2])

        m1 = nn.AvgPool2d((kernel_y[0], kernel_x[0]))
        m2 = nn.AvgPool2d((kernel_y[0], kernel_x[1]))
        m3 = nn.AvgPool2d((kernel_y[1], kernel_x[0]))
        m4 = nn.AvgPool2d((kernel_y[1], kernel_y[1]))

        tl1x = kernel_x[2]
        tl2x = self.x - kernel_x[3]
        tl1y = kernel_y[2]
        tl2y = self.y - kernel_y[3]

        sl1x = kernel_x[2] * kernel_x[0]
        sl2x = x.shape[3] - (kernel_x[3] * kernel_x[0])
        sl1y = kernel_y[2] * kernel_y[0]
        sl2y = x.shape[2] - (kernel_y[3] * kernel_y[0])

        result[:, :, :tl1y, :tl1x] = m1(x[:, :, :sl1y, :sl1x])
        result[:, :, :tl1y, tl1x:tl2x] = m2(x[:, :, :sl1y, sl1x:sl2x])
        result[:, :, :tl1y, tl2x:] = m1(x[:, :, :sl1y, sl2x:])

        result[:, :, tl1y:tl2y, :tl1x] = m3(x[:, :, sl1y:sl2y, :sl1x])
        result[:, :, tl1y:tl2y, tl1x:tl2x] = m4(x[:, :, sl1y:sl2y, sl1x:sl2x])
        result[:, :, tl1y:tl2y, tl2x:] = m3(x[:, :, sl1y:sl2y, sl2x:])

        result[:, :, tl2y:, :tl1x] = m1(x[:, :, sl2y:, :sl1x])
        result[:, :, tl2y:, tl1x:tl2x] = m2(x[:, :, sl2y:, sl1x:sl2x])
        result[:, :, tl2y:, tl2x:] = m1(x[:, :, sl2y:, sl2x:])

        return result


class NeuralNetwork(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.ret_sigmoid = nn.Sigmoid()
        self.layer_input = nn.Sequential(nn.AdaptiveMaxPool2d(120))
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 1), nn.Conv2d(32, 32, 3), nn.BatchNorm2d(32),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(3))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(3))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.LeakyReLU(),
                                    nn.MaxPool2d(3))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.LeakyReLU(),
                                    nn.Flatten())
        self.layer_output = nn.Sequential(
            nn.Linear(2304, 2304), nn.BatchNorm1d(2304), nn.LeakyReLU(), nn.Linear(2304, 2304),
            nn.BatchNorm1d(2304), nn.LeakyReLU(), nn.Linear(2304, 2))

    def forward(self, x):
        x = self.layer_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer_output(x) if self.training else self.ret_sigmoid(self.layer_output(x))
        return x

    def train_model(self, train_data_loader, test_data_loader, num_epochs, learning_rate, model_path, csv_file_name):

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)

        n_total_steps = len(train_data_loader)
        for epoch in range(num_epochs):
            self.train(True)
            for i, (images, labels) in enumerate(train_data_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                output = self(images)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % int(n_total_steps/2) == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss:{loss.item():.4f}")
            print()

            if (epoch + 1) % 10 == 0:
                torch.save(self.state_dict(), f'{model_path}/epoch_{epoch + 1}.pth')
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.eval()
                print("\nModel evaluation on train data")
                tr_loss, tr_acc, tr_auc = self.evaluate_model(train_data_loader)
                print("\nModel evaluation on testing data")
                te_loss, te_acc, te_auc = self.evaluate_model(test_data_loader)
                with open(model_path + "/" + csv_file_name, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, f'{tr_loss:.4f}', f'{te_loss:.4f}',
                                     f'car: {tr_auc[0]:.4f}, empty: {tr_auc[1]:.4f}',
                                     f'car: {te_auc[0]:.4f}, empty: {te_auc[1]:.4f}',
                                     f'{tr_acc:.4f}', f'{te_acc:.4f}'])

        print("Finished Training")

    def evaluate_model(self, data_loader: DataLoader, plot=False):
        with torch.no_grad():
            loss = 0
            n_class_correct = {}
            n_class_samples = {}
            criterion = nn.CrossEntropyLoss()
            class_names = list(data_loader.dataset.classes.keys())
            class_matrix = np.identity(len(class_names))

            for key in class_names:  # initialize class value dict to 0
                n_class_correct[key] = 0  # n_class_correct['car'], n_class_correct['empty']
                n_class_samples[key] = 0

            # creates lists that contains empty list for each class
            all_predictions = []
            all_labels = []

            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                predicted = self(images)
                loss += criterion(predicted, labels)
                predicted = predicted.cpu().numpy()

                for i in range(len(labels)):
                    curr_class = class_names[labels[i]]
                    n_class_samples[curr_class] += 1
                    n_class_correct[curr_class] += 1 if predicted[i].argmax() == labels[i] else 0

                all_predictions.extend(predicted)
                all_labels.extend([class_matrix[i] for i in labels])

            fpr, tpr, roc_auc = count_roc_auc(len(class_names), np.array(all_predictions), np.array(all_labels), plot)

            loss = loss/len(data_loader)
            print(f"Loss of the net: {loss}")
            acc = 100.0 * sum(n_class_correct.values()) / len(data_loader.dataset)
            print(f"Accuracy of the net: {acc}%")

            for i in class_names:
                class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f"Accuracy of {i}: {class_acc}%, ({n_class_correct[i]}/{n_class_samples[i]})")
            return loss, acc, roc_auc
