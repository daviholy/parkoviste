from pathlib import Path
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
from torch import nn
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default='dataset',
                    help='directory where the dataset is stored')
args = parser.parse_args()


class DatasetCreator():
    def __init__(self, annotations_file=None, img_dir=None, transform=None, target_transform=None):
        self.labels = sorted(Path(f"{args.directory}/labels").glob("*.json"))
        self.img_dir = (f"{args.directory}/photos")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        jfile = None
        print(self.labels[idx])
        with open(self.labels[idx], 'r') as j:
            jfile = json.load(j)
        img = read_image(
            f'{args.directory}/photos/{jfile["task"]["data"]["image"].split("/")[-1]}', ImageReadMode.RGB)
        label = jfile["result"][0]["value"]["choices"][0]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

class neuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_input = nn.Sequential(nn.AdaptiveMaxPool2d(120))
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 1), nn.Conv2d(
            32, 32, 3), nn.BatchNorm2d(1), nn.ReLU(), nn.MaxPool2d(3))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.Conv2d(
            64, 64, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.Conv2d(
            128, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(3))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 1), nn.Conv2d(
            256, 256, 3), nn.BatchNorm2d(256), nn.ReLU(), nn.Flatten())
        self.layer_output = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm2d(256), nn.Sigmoid())

    def forward(x,self):
       x = self.layer_input(x)
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)
       return (self.layer_output(x))


if __name__ == "__main__":
    data = DatasetCreator(transform=nn.Sequential(
        transforms.Grayscale(), transforms.RandomEqualize(p=1)))
    img = data[105][0].squeeze()
    plt.imshow(img, cmap='gray')
    plt.show()
