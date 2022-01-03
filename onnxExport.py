from NN.NeuralNetwork import NeuralNetwork
from NN.DatasetCreator import DatasetCreator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms
from torch.nn import Sequential
import argparse
from common import Common
from torch import load
from torch.onnx import export


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='./dataset',
                        help='directory where the dataset is stored')
    parser.add_argument('-em', '--existing_model_path', type=str, default='./model/model_b64_e25.pth',
                        help='path to existing model that you wish to load, if empty model will not be loaded')
    parser.add_argument('-s', '--save_path', type=str, default='model/export.onnx',
                        help='path to where store the exported model')
    args = Common.commonArguments(parser)
    Common.args = args

    # load the pytorch model
    model = NeuralNetwork()
    model.load_state_dict(load(args.existing_model_path, map_location='cpu'))
    model.eval()

    #load the example data
    labels = {"car": 0, "empty": 1}
    batch_size = 96
    trans =  Sequential(transforms.Grayscale())

    data = DatasetCreator(labels, f'{args.directory}/testing/photos', f'{args.directory}/testing/labels', transform=trans)
    loader = DataLoader(data, batch_size=batch_size, sampler=RandomSampler(data_source=data),
                             collate_fn=Common._collate_fn_pad)
    batch, label=next(iter(loader))

    #exporting the model
    export(model, batch, args.save_path, export_params = True, opset_version=13, do_constant_folding = True, input_names= ['input'], output_names= ['output'],
     dynamic_axes={'input': {0: "batch_size", 2: 'width', 3: 'height'}})