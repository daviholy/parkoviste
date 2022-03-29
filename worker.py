"the backround worker for parsing actual information and storing into redis server"

import redis, yaml, json, torch
import datetime
import argparse
from common import Common
from yaml.loader import SafeLoader
from cv2 import VideoCapture, cvtColor, COLOR_BGR2GRAY, equalizeHist, putText, FONT_HERSHEY_DUPLEX, rectangle, imwrite, IMWRITE_JPEG_QUALITY
from torch.utils.data import Dataset
from torch import nn, zeros, tensor
from timeit import default_timer
from time import sleep
from math import floor, ceil
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SequentialSampler
from NN.NeuralNetwork import NeuralNetwork


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
    padded_imgs = zeros(len(batch), 1, max_h, max_w)
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
    def __init__(self, data: list = None, labels: list = None, grayscale=False) -> None:
        self._transform = grayscale
        self._data = data
        self._labels = labels

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self._transform:
            img = cvtColor(self._data[idx], COLOR_BGR2GRAY)
            img = equalizeHist(img)
            return tensor(img).unsqueeze(0), self._labels[idx]
        else:
            return tensor(self._data[idx]), self._labels[idx]

    def setData(self, data: list, labels: list) -> None:
        self._data = data
        self._labels = labels


class CameraCollection:
    """
    utility class which holds all the cameras and aggregate functions which operates on all cameras in collection
    """

    def __init__(self, cameras: list):
        self.collection = []
        for i in range(len(cameras)):
            self.collection.append((Camera(i, cameras[i][0], cameras[i][1])))

    def take_photos(self) -> tuple:
        """
        :return: list of cropped parking places with id
        :rtype: list
        """
        photos = []
        labels = []
        whole_photos = []

        for camera in self.collection:
            ret, photo = camera.take_photo()
            photo_samples, label_cur = camera.cut_photo_samples(photo)
            photos.extend(photo_samples)
            labels.extend(label_cur)
            whole_photos.append(photo)
        return photos, labels, whole_photos


class Camera:
    """
    class which represents connected camera with it's coordinates of places
    """

    def __init__(self, index, address: str, coordinates: str):
        self.coordinates = self._load_json(coordinates)
        self.index = index
        self._connection = VideoCapture(address)

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

    def take_photo(self) -> tuple:
        ret, frame = self._connection.read()  # TODO: implement logging if the connection failed
        return ret, frame

    def cut_photo_samples(self, photo) -> tuple:
        pictures = []
        labels = []
        for place, data in self.coordinates.items():
            x1, y1 = data['coordinates'][0]
            x2, y2 = data['coordinates'][1]
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            pictures.append(photo[y1:y2, x1:x2])
            labels.append(f"{place}-{self.index}")
        return pictures, labels


def save_prediction_img(img, prediction, cameras, save_prediction_path,save_img_path):
    counter = 0
    car_count = 0
    suffix = ".jpg"
    file_name = f'{datetime.datetime.now().strftime("%Y_%b_%d_%H_%M")}_cam'

    for i in range(len(img)):
        imwrite(save_img_path + '/' + file_name + str(i) + suffix , img[i], [int(IMWRITE_JPEG_QUALITY), 85])

    for camera in cameras:
        for place, data in camera.coordinates.items():
            place_type = data["type"]
            coor = data["coordinates"]

            pos = coor[0] if counter < 32 else [coor[0][0], coor[1][1] + 19]
            car_count += int(prediction[counter])

            putText(img=img[camera.index], text=f'{counter}:{prediction[counter]}', org=pos,
                    fontFace=FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
            counter += 1

            # BGR
            rect_color = (0, 255, 0)  # standard - green
            if place_type == "disabled":
                rect_color = (255, 0, 0)  # blue
            elif place_type == "non-standard":
                rect_color = (0, 0, 255)  # red

            rectangle(img[camera.index], coor[0], coor[1], rect_color)


            imwrite(save_prediction_path + '/' + file_name + str(camera.index) + suffix, img[camera.index], [int(IMWRITE_JPEG_QUALITY), 85])
        


parser = argparse.ArgumentParser()
parser.add_argument('-pth', '--save_path', type=str, default='./dataset',
                    help='directory where the whole photo of parking place with prediction will be saved')
args = Common.common_arguments(parser)
Common.args = args

# preloading the config
config = None
# TODO: implement waiting , now if the connections doesn't exist, the script will hang out indefinitely
with open('config.yaml') as f:  # parsing config file
    config = yaml.load(f, Loader=SafeLoader)

# loading camera configurations
connections = CameraCollection(config["cameras"])

data = DatasetCreator(grayscale=True)
loader = DataLoader(data, batch_size=config["batch_size"], sampler=SequentialSampler(data))

model = NeuralNetwork()
model.load_state_dict(
    torch.load(config["model"], map_location=torch.device(config["device"])))
model.eval()

# Creating the redis client connection
conn = None
if config.get("socket") is None:
    if config.get("ip") is None:
        conn = redis.Redis(host='localhost', port=6379, decode_responses=True)
    else:
        conn = redis.Redis(host=config["ip"]["host"], port=config["ip"]["port"], decode_responses=True)
else:
    conn = redis.Redis(unix_socket_path=config["socket"])

"""
the main loop, which will take photos and update the redis database
"""
while True:
    start = default_timer()
    print(str(datetime.datetime.now()))
    # taking photo from cameras and feed them into dataloader
    photos, labels, whole_photos = connections.take_photos()
    predictions = []
    data.setData(photos, labels)
    results = {'timestamp': str(datetime.datetime.now())}
    # interfere with model
    with torch.no_grad():
        try:
            for batch, label in loader:
                pred = model(batch.float()).numpy().argmax(axis=1).tolist()
                results.update(zip(label, pred))
                predictions.extend(pred)
        except StopIteration:
            pass
    conn.hset('parking', mapping=results)

    if config["save"]["enabled"]:
        save_prediction_img(whole_photos, predictions, connections.collection, config["save"]["predicted_path"], config["save"]["clean_img_path"])

    end = default_timer()
    if (config["interval"] - (end - start)) > 0:
        sleep(config["interval"] - (end - start))
