from pathlib import Path
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
from torch import nn, tensor, Tensor, device, cuda
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default='./dataset/split/',
                    help='directory where the dataset is stored')
args = parser.parse_args()

training_dir = {'labels': f'{args.directory}/training/labels/',
                'photos': f'{args.directory}/training/photos/'}
testing_dir = {'labels': f'{args.directory}/testing/labels/',
               'photos': f'{args.directory}/testing/photos/'}

dest_dirs = {'training': training_dir, 'testing': testing_dir}


class DatasetCreator():
    def __init__(self, dataset_type, annotations_file=None, img_dir=None, transform=None, target_transform=None):
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


def collate_fn_pad(batch):
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

    padded_imgs = []
    # padding
    for img in imgs:
        pad_h = max_h - img[0].size(0)
        pad_w = max_w - img[0].size(1)

        pad_l = int(pad_w/2)  # left
        pad_r = pad_w - pad_l  # right
        pad_t = int(pad_h/2)  # top
        pad_b = pad_h - pad_t  # bottom
        pad = nn.ZeroPad2d((pad_l, pad_r, pad_t, pad_b))
        padded_imgs.append(pad(img[0]))

    return padded_imgs, labels


def split_dataset(train_data, test_data, batch_size=1, shuffle=True, random_seed=42):
    """
    Splits dataset to training and testing set by given ratio.
    :param test_data: dataset with testing data from DataCreator
    :param train_data: dataset with training data from DataCreator
    :param batch_size: size of the batch
    :param random_seed: seed for random shuffling
    :param shuffle: bool - decides if dataset will be shuffled
    :return:
    :rtype: training and testing dataset
    """

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    train_indices, test_indices = list(range(train_data_size)), list(range(test_data_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(test_indices)
        np.random.shuffle(train_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                              collate_fn=collate_fn_pad, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler,
                             collate_fn=collate_fn_pad, drop_last=True)

    return train_loader, test_loader


def test_data_loaders(train_data, test_data):
    """
    Test function for data_loaders.
    :return:
    """
    train_loader, test_loader = split_dataset(train_data, test_data, batch_size=5)
    d_train = {'car': 0, 'empty': 0, 'skipped': 0}
    d_test = {'car': 0, 'empty': 0, 'skipped': 0}
    for batch_index, (imgs, labels) in enumerate(train_loader):
        img = imgs[0]
        for l in labels:
            if len(l) > 0:
                d_train[l] = d_train[l] + 1
            else:
                d_train['skipped'] = d_train['skipped'] + 1
    for batch_index, (imgs, labels) in enumerate(test_loader):
        for l in labels:
            if len(l) > 0:
                d_test[l] = d_test[l] + 1
            else:
                d_test['skipped'] = d_test['skipped'] + 1
    print(f"train_dataset stats: {d_train}, cars in dataset: {d_train['car'] / (d_train['car'] + d_train['empty'])}%")
    print(f"test_dataset stats: {d_test}, cars in dataset: {d_test['car'] / (d_test['car'] + d_test['empty'])}%")


if __name__ == "__main__":
    train_data = DatasetCreator('training', transform=nn.Sequential(
        transforms.Grayscale(), transforms.RandomEqualize(p=1)))
    test_data = DatasetCreator('testing', transform=nn.Sequential(
        transforms.Grayscale(), transforms.RandomEqualize(p=1)))
    img = train_data[105][0].squeeze()
    test_data_loaders(train_data=train_data, test_data=test_data)
    plt.imshow(img, cmap='gray')
    plt.show()
