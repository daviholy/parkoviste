from common import Common
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from torch import zeros
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
import argparse
import os
import sys
from sys import exit
from NN.DatasetCreator import *
from NN.NeuralNetwork import *


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

    padded_imgs = zeros(len(batch), 1, max_h, max_w)
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
        # padded_imgs[x] = F.pad(img, (pad_l, pad_r, pad_t, pad_b), value=255)  # test of white padding (seems same)
    return padded_imgs, torch.stack(labels)

def test_data_loaders(train_loader, test_loader):
    """
    Test function for data_loaders. Prints datasets statistics.
    :return:
    """
    cls = ['empty', 'car']
    d_train = {'car': 0, 'empty': 0, 'skipped': 0}
    d_test = {'car': 0, 'empty': 0, 'skipped': 0}
    for batch_index, (imgs, labels) in enumerate(train_loader):
        for l in labels:
            idx = int(l.item())
            if len(l) > 0:
                d_train[cls[idx]] = d_train[cls[idx]] + 1
            else:
                d_train['skipped'] = d_train['skipped'] + 1
    for batch_index, (imgs, labels) in enumerate(test_loader):
        img = imgs[0]
        for l in labels:
            idx = int(l.item())
            if len(l) > 0:
                d_test[cls[idx]] = d_test[cls[idx]] + 1
            else:
                d_test['skipped'] = d_test['skipped'] + 1
    print(f"train_dataset stats: {d_train}, cars in dataset: "
          f"{round(d_train['car'] / (d_train['car'] + d_train['empty']) * 100, 2)}%")
    print(f"test_dataset stats: {d_test}, cars in dataset: "
          f"{round(d_test['car'] / (d_test['car'] + d_test['empty']) * 100, 2)}%")
    print(img)
    print(img.shape)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='./dataset/split',
                        help='directory where the dataset is stored')
    parser.add_argument('-train', '--train_mode', type=bool, default=False,
                        help='decides if model will be trained or only evaluated')
    parser.add_argument('-em', '--existing_model_path', type=str, default='',
                        help='path to existing model that you wish to load, if empty model will not be loaded')
    parser.add_argument('-sm', '--save_model_path', type=str, default='',
                        help='path with name of the model that you wish to save, if empty model will not be saved')
    args = Common.commonArguments(parser)
    Common.args = args

    training_dir = {'labels': f'{args.directory}/training/labels',
                    'photos': f'{args.directory}/training/photos'}
    testing_dir = {'labels': f'{args.directory}/testing/labels',
                   'photos': f'{args.directory}/testing/photos'}

    dest_dirs = {'training': training_dir, 'testing': testing_dir}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    num_epochs = 16
    learning_rate = 0.0025
    batch_size = 96
    labels = {"car": 0, "empty": 1}

    trans = nn.Sequential(
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(),
        transforms.RandomEqualize(p=1))

    train_data = DatasetCreator(labels, training_dir['photos'], training_dir['labels'], transform=trans)
    test_data = DatasetCreator(labels, testing_dir['photos'], testing_dir['labels'], transform=trans)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=RandomSampler(data_source=train_data),
                              collate_fn=_collate_fn_pad)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=RandomSampler(data_source=test_data),
                             collate_fn=_collate_fn_pad)

    test_data_loaders(train_loader, test_loader)

    model = NeuralNetwork(device).to(device)

    # Check input arguments
    if len(args.existing_model_path) > 0:
        if not os.path.isfile(args.existing_model_path):
            exit("invalid path to model you wish to load")
        model.load_state_dict(torch.load(args.existing_model_path))

    if len(args.save_model_path) > 0:
        if not os.path.isdir('/'.join(os.path.split(args.save_model_path)[:-1])):
            exit("invalid path to save model")

    if args.train_mode:
        model.train_model(train_loader, test_loader, num_epochs, learning_rate)
        if len(args.save_model_path) > 0:
            torch.save(model.state_dict(), args.save_model_path)
    else:
        model.eval()
        model.evaluate_model(test_loader, 0.5)
