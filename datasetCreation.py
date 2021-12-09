import os.path
import torch
from sys import exit
from pathlib import Path
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
from torch import nn
from torch import tensor
from torch import zeros
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
import json
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default='./dataset/split',
                    help='directory where the dataset is stored')
parser.add_argument('--DEBUG', type=bool, default=False,
                    help='decides if script will run in debug mode (prints to stdout)')
args = parser.parse_args()

training_dir = {'labels': f'{args.directory}/training/labels',
                'photos': f'{args.directory}/training/photos'}
testing_dir = {'labels': f'{args.directory}/testing/labels',
               'photos': f'{args.directory}/testing/photos'}

dest_dirs = {'training': training_dir, 'testing': testing_dir}


class DatasetCreator(Dataset):
    def __init__(self, dataset_type, annotations_file=None, transform=None, target_transform=None):

        if not os.path.isdir(Path(dest_dirs[dataset_type]['labels'])):
            exit("Not a valid directory")

        self.labels = sorted(Path(dest_dirs[dataset_type]['labels']).glob("*.json"))
        self.dataset_type = dataset_type
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        jfile = None
        with open(self.labels[idx], 'r') as j:
            jfile = json.load(j)
        img = read_image(
            f'{dest_dirs[self.dataset_type]["photos"]}/{jfile["task"]["data"]["image"].split("/")[-1]}',
            ImageReadMode.RGB)
        if len(jfile["result"]) != 0:
            label = jfile["result"][0]["value"]["choices"][0]
        else:
            return img, []
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img.float(), tensor(1) #TODO: implement transformation text to tensor (now its only returning 1 always)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_input = nn.Sequential(nn.AdaptiveMaxPool2d(120))
        self.conv = nn.Conv2d(1, 32, 1)
        self.conv2 =nn.Conv2d(32, 32, 3)
        self.batchnorm = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3)
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 1), nn.Conv2d(
            32, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.Conv2d(
            64, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.Conv2d(
            128, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(3))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Flatten())
        self.lin1 = nn.Identity()
        self.lin2 = nn.LazyLinear(1)
        self.sig = nn.Sigmoid()
        self.layer_output = nn.Sequential(
            nn.Identity(), nn.BatchNorm1d(2304), nn.ReLU(), nn.Identity(), nn.BatchNorm1d(2304), nn.ReLU(),  nn.LazyLinear(1), nn.Sigmoid())

    def forward(self, x):
        x = self.layer_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer_output(x)


def _collate_fn_pad(batch):
    """
    Takes images as tensors and labels as input, finds highest and widest size of an image than pad smaller images.
    :param batch: list of images and labels [img_as_tensor, 'label', ....]
    :return: returns tuple where ([padded_img_as_tensors,..], ('label',..))
    :rtype: (list, tuple)
    """
    imgs, labels = zip(*batch)
    # get max width and height
    h, w = zip(*[list(t[0].size()) for t in imgs])
    max_h, max_w = max(h), max(w)

    padded_imgs = zeros(len(batch),1,max_h,max_w)
    # padding
    for x in range(len(batch)):
        img = batch[x][0]
        pad_h = max_h - img[0].size(0)
        pad_w = max_w - img[0].size(1)

        pad_l = int(pad_w / 2)  # left
        pad_r = pad_w - pad_l  # right
        pad_t = int(pad_h / 2)  # top
        pad_b = pad_h - pad_t  # bottom
        pad = nn.ZeroPad2d((pad_l, pad_r, pad_t, pad_b))
        padded_imgs[x] = pad(img)

    return padded_imgs, labels


def debug(func):
    def inner(*arg):
        if args.DEBUG:
            func(*arg)

    return inner


@debug
def test_data_loaders(train_loader, test_loader):
    """
    Test function for data_loaders. Prints datasets statistics.
    :return:
    """
    d_train = {'car': 0, 'empty': 0, 'skipped': 0}
    d_test = {'car': 0, 'empty': 0, 'skipped': 0}
    for batch_index, (imgs, labels) in enumerate(train_loader):
        for l in labels:
            if len(l) > 0:
                d_train[l] = d_train[l] + 1
            else:
                d_train['skipped'] = d_train['skipped'] + 1
    for batch_index, (imgs, labels) in enumerate(test_loader):
        img = imgs[0]
        for l in labels:
            if len(l) > 0:
                d_test[l] = d_test[l] + 1
            else:
                d_test['skipped'] = d_test['skipped'] + 1
    print(f"train_dataset stats: {d_train}, cars in dataset: "
          f"{round(d_train['car'] / (d_train['car'] + d_train['empty']) * 100, 2)}%")
    print(f"test_dataset stats: {d_test}, cars in dataset: "
          f"{round(d_test['car'] / (d_test['car'] + d_test['empty']) * 100, 2)}%")
    plt.imshow(img.squeeze(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    batch_size = 5

    train_data = DatasetCreator('training', transform=nn.Sequential(
        transforms.Grayscale(), transforms.RandomEqualize(p=1)))
    test_data = DatasetCreator('testing', transform=nn.Sequential(
        transforms.Grayscale(), transforms.RandomEqualize(p=1)))

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=RandomSampler(data_source=train_data),
                              collate_fn=_collate_fn_pad)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=RandomSampler(data_source=test_data),
                             collate_fn=_collate_fn_pad)

    test_data_loaders(train_loader, test_loader)

    training = NeuralNetwork().to("cpu")

    #1 epoch
    for (data, label) in train_loader:
        print(training(data))