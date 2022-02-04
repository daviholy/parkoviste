import random
from common import Common
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
import argparse
import os
import csv
from sys import exit
from datetime import datetime
from NN.DatasetCreator import *
from NN.NeuralNetwork import *
import matplotlib.pyplot as plt


"""In this script NN model is trained."""


@Common.debug("model statistics")
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
                d_train[cls[idx]] += 1
            else:
                d_train['skipped'] += 1
    for batch_index, (imgs, labels) in enumerate(test_loader):
        img = imgs[0]
        for l in labels:
            idx = int(l.item())
            if len(l) > 0:
                d_test[cls[idx]] += 1
            else:
                d_test['skipped'] += 1
    print(f"train_dataset stats: {d_train}, cars in dataset: "
          f"{round(d_train['car'] / (d_train['car'] + d_train['empty']) * 100, 2)}%")
    print(f"test_dataset stats: {d_test}, cars in dataset: "
          f"{round(d_test['car'] / (d_test['car'] + d_test['empty']) * 100, 2)}%")
    print(img)
    print(img.shape)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.show()


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1., p=0.5):
        super().__init__()
        self.std = std
        self.mean = mean
        self.p = p

    def forward(self, input_tensor):
        res = input_tensor
        if random.random() >= self.p:
            res = input_tensor + torch.randn(input_tensor.size()) * self.std + self.mean
        return res.type(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='./dataset',
                        help='directory where the dataset is stored')
    parser.add_argument('-train', '--train_mode', type=bool, default=False,
                        help='decides if model will be trained or only evaluated')
    parser.add_argument('-em', '--existing_model_path', type=str, default='',
                        help='path to existing model that you wish to load, if empty model will not be loaded')
    parser.add_argument('-sm', '--save_model_path', type=str, default='',
                        help='path with name of the model that you wish to save, if empty model will not be saved')
    args = Common.common_arguments(parser)
    Common.args = args

    training_dir = {'labels': f'{args.directory}/training/labels',
                    'photos': f'{args.directory}/training/photos'}
    testing_dir = {'labels': f'{args.directory}/testing/labels',
                   'photos': f'{args.directory}/testing/photos'}

    dest_dirs = {'training': training_dir, 'testing': testing_dir}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 200
    learning_rate = 0.00001
    batch_size = 32
    labels = {"empty": 0, "car": 1}

    train_trans = nn.Sequential(
        transforms.ColorJitter(hue=0.3, saturation=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        AddGaussianNoise(mean=0.0, std=0.5, p=0.2),
        transforms.RandomErasing(p=0.2, scale=(0.08, 0.08), ratio=(1, 2))
    )

    train_data = DatasetCreator(labels, training_dir['photos'], training_dir['labels'], transform=train_trans)
    test_data = DatasetCreator(labels, testing_dir['photos'], testing_dir['labels'])

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=RandomSampler(data_source=train_data),
                              collate_fn=Common.collate_fn_pad)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=RandomSampler(data_source=test_data),
                             collate_fn=Common.collate_fn_pad)

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
        save_model_path = '/'.join(args.save_model_path.split('/')[:-1])
        csv_file_name = "model_info.csv"
        with open(save_model_path + '/' + csv_file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch_num', 'train_loss', 'test_loss', 'train_auc', 'test_auc', 'train_acc', 'test_acc',
                             '', f'date_time: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}'])
        model.train_model(train_loader, test_loader, num_epochs, learning_rate, save_model_path, csv_file_name)
        if len(args.save_model_path) > 0:
            torch.save(model.state_dict(), args.save_model_path)
    else:
        model.eval()
        model.evaluate_model(test_loader, plot=True)
