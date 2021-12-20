import torch
from torchvision import transforms
from torch import nn
from torch import zeros
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader
import argparse
import matplotlib.pyplot as plt
from NN.DatasetCreator import DatasetCreator
from NN.NeuralNetwork import NeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default='./dataset',
                    help='directory where the dataset is stored')
parser.add_argument('--DEBUG', type=bool, default=False,
                    help='decides if script will run in debug mode (prints to stdout)')
args = parser.parse_args()

training_dir = {'labels': f'{args.directory}/training/labels',
                'photos': f'{args.directory}/training/photos'}
testing_dir = {'labels': f'{args.directory}/testing/labels',
               'photos': f'{args.directory}/testing/photos'}

dest_dirs = {'training': training_dir, 'testing': testing_dir}


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
    return padded_imgs, torch.stack(labels)


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 8
    learning_rate = 0.001
    batch_size = 5
    labels = {"car": [1.,0.], "empty": [0.,1.]
    }
    #train_data = DatasetCreator(training_dir['photos'], training_dir['labels'], transform=nn.Sequential(
    #    transforms.Grayscale(), transforms.RandomEqualize(p=1)))
    test_data = DatasetCreator(labels,testing_dir['photos'], testing_dir['labels'], transform=nn.Sequential(
        transforms.Grayscale(), transforms.RandomEqualize(p=1)))

    #train_loader = DataLoader(train_data, batch_size=batch_size, sampler=RandomSampler(data_source=train_data),
    #                          collate_fn=_collate_fn_pad)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=RandomSampler(data_source=test_data),
                             collate_fn=_collate_fn_pad)

    # test_data_loaders(train_loader, test_loader)


    model = NeuralNetwork(device)

   # model.load_state_dict(torch.load("../model/model2.pth"))  # Toto je nacitani jiz existujiciho modelu
    model.eval()

    model.evaluate_model(test_loader)
    # train_model()

