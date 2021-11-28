from pathlib import Path
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
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
        self.img_dir = f"{args.directory}/photos"
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
        if len(jfile["result"]) != 0:
            label = jfile["result"][0]["value"]["choices"][0]
        else:
            return [], []
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class NeuralNetwork(nn.Module):
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

    def forward(self, x):
        x = self.layer_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer_output(x)


def split_dataset(dataset, train_split_ratio, batch_size=1, shuffle=True, random_seed=42):
    """
    Splits dataset to training and testing set by given ratio.
    :param batch_size: size of the batch
    :param random_seed: seed for random shuffling
    :param shuffle: bool - decides if dataset will be shuffled
    :param dataset: pytorch dataset
    :param train_split_ratio: float in range 0-1 that determines ratio of slit between train and test dataset
    :return:
    :rtype: training and testing dataset
    """

    data_size = len(dataset)
    indices = list(range(data_size))
    split = int(np.floor(train_split_ratio * data_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    test_indices, train_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader


def test_data_loaders():
    """
    Test function for data_loaders.
    :return:
    """
    # FIXME: pokud zadame batch_size > 1 tak loader kontroluje velikost obrazku v kazde batch a
    #  hazi error pri rozdilne velikosti obrazku
    train_loader, test_loader = split_dataset(data, train_split_ratio=.7)
    d_train = {'car': 0, 'empty': 0, 'skipped': 0}
    d_test = {'car': 0, 'empty': 0, 'skipped': 0}
    for batch_index, (imgs, labels) in enumerate(train_loader):
        if len(labels) > 0:
            d_train[labels[0]] = d_train[labels[0]] + 1
        else:
            d_train['skipped'] = d_train['skipped'] + 1
    for batch_index, (imgs, labels) in enumerate(test_loader):
        if len(labels) > 0:
            d_test[labels[0]] = d_test[labels[0]] + 1
        else:
            d_test['skipped'] = d_test['skipped'] + 1
    print(f"train_dataset stats: {d_train}, cars in dataset: {d_train['car']/(d_train['car']+d_train['empty'])}%")
    print(f"test_dataset stats: {d_test}, cars in dataset: {d_test['car']/(d_test['car']+d_test['empty'])}%")


if __name__ == "__main__":
    data = DatasetCreator(transform=nn.Sequential(
        transforms.Grayscale(), transforms.RandomEqualize(p=1)))
    img = data[105][0].squeeze()
    test_data_loaders()
    plt.imshow(img, cmap='gray')
    plt.show()
