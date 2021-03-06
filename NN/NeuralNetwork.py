import numpy as np
from torch import nn, zeros, tensor, Tensor
from torch.optim import Adam
import torch
from torch.utils.data.dataloader import DataLoader
from NN.ModelStatistics import *
from math import floor, ceil
from torch.nn import ConstantPad2d
import numpy as np

class MyAdaptiveAvgPool2d():
    def __init__(self, x,y):
        super().__init__()
        self.y = y
        self.x = x

    def kernel(self,x_out,x_in):
            if x_out == x_in:
                return(tensor(1),tensor(1),0,0)
            #calculate how many times we need to repeat the bigger kernel
            f = floor(x_in/x_out)
            c = ceil(x_in/x_out)
            n = ((f*x_out) - x_in)/(f-c)
            if n%2 == 0 :
                return c,f,int(n/2),int(n/2)
            else:
                return c,f,int((n+1)/2),int((n-1)/2)
    
    def forward(self, x) -> Tensor: 

        #maxpooling the bigger dimensions
        result = zeros(x.shape[0],x.shape[1],self.y,self.x)
        kernel_x = self.kernel(self.x,x.shape[3])
        kernel_y = self.kernel(self.y,x.shape[2])

        m1 = nn.AvgPool2d((kernel_y[0],kernel_x[0]))
        m2 = nn.AvgPool2d((kernel_y[0],kernel_x[1]))
        m3 = nn.AvgPool2d((kernel_y[1],kernel_x[0]))
        m4 = nn.AvgPool2d((kernel_y[1],kernel_y[1]))

        tl1x = kernel_x[2]
        tl2x = self.x-kernel_x[3]
        tl1y = kernel_y[2]
        tl2y = self.y-kernel_y[3]

        sl1x = kernel_x[2] * kernel_x[0]
        sl2x = x.shape[3] - (kernel_x[3] * kernel_x[0])
        sl1y = kernel_y[2] * kernel_y[0]
        sl2y = x.shape[2] - (kernel_y[3] * kernel_y[0])


        result[:,:,:tl1y,:tl1x] = m1(x[:,:,:sl1y,:sl1x])
        result[:,:,:tl1y,tl1x:tl2x] = m2(x[:,:,:sl1y,sl1x:sl2x])
        result[:,:,:tl1y,tl2x:] = m1(x[:,:,:sl1y,sl2x:])

        result[:,:,tl1y:tl2y,:tl1x] = m3(x[:,:,sl1y:sl2y,:sl1x])
        result[:,:,tl1y:tl2y,tl1x:tl2x] = m4(x[:,:,sl1y:sl2y,sl1x:sl2x])
        result[:,:,tl1y:tl2y,tl2x:] = m3(x[:,:,sl1y:sl2y,sl2x:])

        result[:,:,tl2y:,:tl1x] = m1(x[:,:,sl2y:,:sl1x])
        result[:,:,tl2y:,tl1x:tl2x] = m2(x[:,:,sl2y:,sl1x:sl2x])
        result[:,:,tl2y:,tl2x:] = m1(x[:,:,sl2y:,sl2x:])



        return result

class NeuralNetwork(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.layer_input = nn.Sequential(nn.AdaptiveMaxPool2d(120))
        self.Conv1 = nn.Conv2d(1, 32, 1)
        self.Conv2 = nn.Conv2d(32, 32, 3)
        self.norm1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.3)
        self.pool = nn.MaxPool2d(3)
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 1), nn.Conv2d(32, 32, 3), nn.BatchNorm2d(32), nn.Dropout(p=0.3),nn.LeakyReLU(),
                                    nn.MaxPool2d(3))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64), nn.Dropout(p=0.3), nn.LeakyReLU(),
                                    nn.MaxPool2d(3))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.Dropout(p=0.3), nn.LeakyReLU(),
                                    nn.MaxPool2d(3))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.Dropout(p=0.3), nn.LeakyReLU(), nn.Flatten())
        self.layer_output = nn.Sequential(
            nn.Linear(2304,2304), nn.BatchNorm1d(2304), nn.LeakyReLU(), nn.Linear(2304,2304), nn.BatchNorm1d(2304), nn.LeakyReLU(),  nn.Linear(2304,2))

       
    def forward(self, x):
        x = self.layer_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer_output(x)

    def train_model(self, train_data_loader, test_data_loader, num_epochs, learning_rate):

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)

        accuracy_stats = {'train': [],
                            "test": []}

        loss_stats = {'train': [],
                    "test": []}
        n_total_steps = len(train_data_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_data_loader):

                images, labels = images.to(self.device), labels.to(self.device)

                output = self(images)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")
                print()
                print("Model evaluation on testing data")
                accuracy_stats["test"], loss_stats["test"] = self.evaluate_model(test_data_loader, 0.1)
                print()
                print("Model evaluation on train data")
                accuracy_stats["train"], loss_stats["train"] = self.evaluate_model(train_data_loader, 0.1)
                print()

        print("Finished Training")
        return accuracy_stats, loss_stats


    def evaluate_model(self, data_loader: DataLoader, plot= False ):
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
                predicted = predicted.numpy()

                for i in range(len(predicted)):
                    n_class_samples[data_loader.dataset.labels[int(labels[i])]] +=1
                    if predicted[i].argmax() == labels[i]:
                        n_class_correct[data_loader.dataset.labels[int(labels[i])]] +=1

                all_predictions.extend(predicted)
                all_labels.extend([class_matrix[i] for i in labels])

            fpr, tpr, roc_auc = count_roc_auc(len(class_names), np.array(all_predictions), np.array(all_labels), plot)

            print(f"Loss of the net: {loss}")
            acc = 100.0 * sum(n_class_correct.values()) / len(data_loader.dataset)
            print(f"Accuracy of the net: {acc}%")

            for i in data_loader.dataset.classes:
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f"Accuracy of {i}: {acc}%")

            print(f'total number of samples: {n_class_samples}')
            print(f'total number of correct guesses: {n_class_correct}')
            return acc, loss