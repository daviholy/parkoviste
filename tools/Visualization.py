if __name__ == "__main__":
    from common import init
    init()

import torch
from torch import nn
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
from NN.NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
import argparse

"""
Script that visualize CNN convolutional layers with given image. Then visualization is saved to jpg file.
"""

def set_model(classes, device, created_model_path):
    """
    Load CNN model and evaluate it.

    :param classes: Classes
    :param device: CPU or GPU
    :param created_model_path: Path to trained CNN model
    :return: Evaluated CNN model
    """
    model = NeuralNetwork(classes, device)
    model.load_state_dict(torch.load(created_model_path, map_location=device))
    model.eval()
    return model


def image_transform(img_path):
    """
    Load an image and transform it.

    :param img_path: Path to image
    :return: Transformed image
    """
    img = read_image(img_path, ImageReadMode.RGB)
    transform = torch.nn.Sequential(transforms.Grayscale(), transforms.RandomEqualize(p=1))
    img = transform(img).float()
    image = img.unsqueeze(0)
    image = image.to(device)

    return image


def get_conv2_layers(model):
    """
    Get all convolution layers from CNN model and count a number of layers.

    :param model: CNN model
    :return: Number of convolutional layers and array with that layers
    """
    conv_layers = []

    model_children = list(model.children())

    num_of_conv2_layers = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                if type(model_children[i][j]) == nn.Conv2d:
                    num_of_conv2_layers += 1
                    conv_layers.append(model_children[i][j])

    return num_of_conv2_layers, conv_layers


def eval_conv2_layers(conv_layers, image):
    """
    Use each CNN convolutional layers on image. Use SUM on each feature map of images and extract gray scale to convert
    images to lower dimension.

    :param conv_layers: Array of convolutional layers
    :param image: Transformed image
    :return: Array of visualized images
    """
    outputs = []
    for layer in conv_layers:
        image = layer(image)
        outputs.append(image)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    return processed


def debug_print(num_of_conv2_layers, outputs, processed):
    """
    Print every important information about script for debug.

    :param num_of_conv2_layers: Number of layers
    :param outputs: Every layer used on a image
    :param processed: Visualized images
    :return:
    """
    for feature_map in outputs:
        print(f"Feature map output: {feature_map.shape}")

    for fm in processed:
        print(f"Processed shape: {fm.shape}")

    print(f"Total convolutional layers: {num_of_conv2_layers}")


def save_figure(processed, num_of_conv2_layers, save_path):
    """
    Save figure of visualized images.
    
    :param processed: Visualized images
    :param num_of_conv2_layers: Number of layers
    :param save_path: Path where figure will be saved
    """
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(1, num_of_conv2_layers, i+1)
        img_plot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(f"K{i+1}", fontsize=30)
    plt.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualization of CNN model')
    parser.add_argument('path_to_model', type=str, help='Path to model.')
    parser.add_argument('path_to_image', type=str, help='Path to image that will be visualized')
    parser.add_argument('save_path', type=str, help='Path where will be figure saved')
    args = parser.parse_args()

    classes = ('empty', 'car')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = set_model(classes, device, created_model_path=args.path_to_model)
    image = image_transform(img_path=args.path_to_image)
    num_of_conv2_layers, conv_layers = get_conv2_layers(model)
    processed = eval_conv2_layers(conv_layers, image)
    save_figure(processed, num_of_conv2_layers, save_path=args.save_path)
