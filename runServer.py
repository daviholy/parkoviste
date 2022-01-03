# import pyyaml module
import collections
import yaml
from multiprocessing import Manager
from flask import Flask
from yaml.loader import SafeLoader
from cv2 import VideoCapture, cvtColor, COLOR_BGR2GRAY, IMREAD_COLOR, imdecode
from torch.utils.data import Dataset
from torch import nn , zeros, tensor
import torch
import json
from timeit import default_timer
from time import sleep
import numpy as np
from math import floor,ceil
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SequentialSampler
import sys
from NN.NeuralNetwork import NeuralNetwork
from multiprocessing.shared_memory import ShareableList

app = Flask(__name__)

def _collate_fn_pad(batch):
     """
     Takes images as tensors and labels as input, finds highest and widest size of an image than pad smaller images 
     :param batch: list of images [img_as_tensor, ....]
     :return: returns tuple where [padded_img_as_tensors,..]
     :rtype: list
     """
     imgs, label = zip(*batch)
     # get max width and height
     h, w = zip(*[t.shape for t in imgs])
     max_h, max_w = max(h), max(w)   
     padded_imgs = zeros(len(batch),1,max_h,max_w)
     # padding
     for x in range(len(batch)):
         img = batch[x][0]
         pad_h = max_h - img.shape[0]
         pad_w = max_w - img.shape[1]
         pad_l = ceil(pad_w / 2)  # left
         pad_r = floor(pad_w - pad_l)  # right
         pad_t = ceil(pad_h / 2)  # top
         pad_b = floor(pad_h - pad_t)  # bottom
         pad = nn.ZeroPad2d((pad_l, pad_r, pad_t, pad_b))
         padded_imgs[x] = pad(img)   
     return padded_imgs, label


class DatasetCreator(Dataset):
    def __init__(self, data : list = None,grayscale= False ) -> None:
        self._transform = grayscale
        self._data = data

    def __len__(self):
        return len(self._data)  

    def __getitem__(self, idx) :
        if self._transform:
            return tensor(cvtColor(self._data[idx][0], COLOR_BGR2GRAY )), self._data[idx][1]
        else:
            return tensor(self._data[idx][0]), self._data[idx][1]

    def setData(self, data: list) ->None:
        self._data = data


class CameraCollection():
    """
    utility class which holds all the cameras and aggregate functions which operates on all cameras in collection
    """

    def __init__ (self, cameras : list):
        self.collection = []
        for adress, coordinates in cameras:
            self.collection.append(Camera(adress,coordinates))

    def example_dictionary(self):
        """
        return the example dictionary in which the data will be returned (in three lists first name, second types, third values)
        """
        types = []
        names = []
        for camera in self.collectionn:
            cam = camera.example_dictionary()
            names.extend(cam[0])
            types.extend(cam[1])
        return (names, types, [0] * len(names))

    def take_photos(self) -> list():
        """
        :return: list of cropped parking places with id
        :rtype: list
        """
        photos =[]
        for camera in self.collection:
            photos.extend(camera.take_photo())
        return photos


class Camera():
    """
    class which represents connected camera with it's coordinates of places
    """
    def __init__(self, adress: str , coordinates : str):
        self._connection = VideoCapture(adress)
        self._coordinates = self._load_json(coordinates)

    @staticmethod
    def _load_json(file_path):
        """
        Loads json file as dictionary.
        :param file_path: string with path to file
        :return: dictionary structure with coordinates
        """

        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            exit("File " + file_path + " not found")

    def  example_dictionary(self):
        """
        return the example dictionary in which the data will be returned (in three lists first name, second types, third values)
        """
        types = []
        names = []

        for place, data in self._coordinates.items():
            names.append(place)
            types.append(data["type"])
        return (names, types , [0] * len(names))
        

    def take_photo(self) -> tuple:
        ret, frame = self._connection.read() #TODO: implement logging if the connection failed
        pictures = []
        for place, data in self._coordinates.items():
            x1, y1 = data['coordinates'][0]
            x2, y2 = data['coordinates'][1]
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            pictures.append(((frame[y1:y2, x1:x2]),place))
        return pictures


class ParkingPlace():
    """
    holds the dictionary of actual situation of parking place and can it return in json format
    """

    
    @classmethod
    def setdata(data : dict) -> None:
        """
        set the dict
        """
        ParkingPlace.data = data

    @classmethod
    @app.route('/')
    def to_json() -> str:
        """
        return data in json if they are set, if not return empty string.
        :return: json data
        :rtype: str
        """
        # if ParkingPlace.data == None:
        #     return ""
        # else:
        #     return json.dumps(ParkingPlace.data)
        return json.dumps(dict(parking_place))

    

def active_loop():
    """
    the main loop, which will take photos and update the shared memory
    """
    while(True):
        start = default_timer()
        #taking photo from cameras and feed them into dataloader
        data.setData(connections.take_photos())
        results =[]
        #interfere with model
        with torch.no_grad():
            for batch, label in loader:
                results.extend(model(batch))
        end = default_timer()
        if (end - start) >0:
            sleep(config["interval"])

#global shared data
keys = []
types = []
values = []

#preloading the config
config = None
#TODO: implement waiting , now if the connections doesn't exist, the script will hang out indefinitely
with open('config.yaml') as f: # parsing config file
        config = yaml.load(f, Loader=SafeLoader)

#loading camera configurations
#connections = CameraCollection(config["cameras"])

data = DatasetCreator(grayscale=True)
loader = DataLoader(data, batch_size=64,collate_fn=_collate_fn_pad, sampler=SequentialSampler(data)) # TODO: read batchsize from cfg

model = NeuralNetwork()
model.load_state_dict(torch.load("./model/Mymodel.pth", map_location=torch.device('cpu')))#TODO: parsing model form cfg
model.eval()

dict = connections.example_dictionary()
keys = ShareableList(dict[0])
types = ShareableList(dict[1])
values = ShareableList(dict[2])
active_loop()


        
