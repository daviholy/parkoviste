import torch
from torch import nn
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
import argparse

def set_model(classes, device, created_model_path):
    model = NeuralNetwork(classes, device)
    model.load_state_dict(torch.load(created_model_path, map_location=device))
    model.eval()

    return model


def image_transform(img_path):
    img = read_image(img_path, ImageReadMode.RGB)
    transform = torch.nn.Sequential(transforms.Grayscale(), transforms.RandomEqualize(p=1))
    img = transform(img).float()
    image = img.unsqueeze(0)
    image = image.to(device)

    return image


def get_conv2_layers(model):
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
    for feature_map in outputs:
        print(f"Feature map output: {feature_map.shape}")

    for fm in processed:
        print(f"Processed shape: {fm.shape}")

    print(f"Total convolutional layers: {num_of_conv2_layers}")


def save_figure(processed, num_of_conv2_layers, save_path):
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

    model = set_model(classes, device, created_model_path = args.path_to_model)
    image = image_transform(img_path = args.path_to_image)
    num_of_conv2_layers, conv_layers = get_conv2_layers(model)
    processed = eval_conv2_layers(conv_layers, image)
    save_figure(processed, num_of_conv2_layers, save_path = args.save_path)