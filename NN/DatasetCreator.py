from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import ImageReadMode
from torchvision.io import read_image
from torch import tensor
from os import path
import os
import json
import cv2


class DatasetCreator(Dataset):
    def __init__(self, label_dict: dict, dest_photos="dataset/photos", dest_labels="dataset/labels", transform=None, target_transform=None):
        # print(os.getcwd())
        if not path.isdir(Path(dest_photos)):
            exit("Not a valid image directory")
        if not path.isdir(Path(dest_labels)):
            exit("Not a valid label directory")
        self.classes = label_dict
        self.labels = {value: key for key, value in label_dict.items()}
        self._labels = sorted(Path(dest_labels).glob("*.json"))
        self._transform = transform
        self._target_transform = target_transform
        self._photos = dest_photos

        self._img_label_lst = []
        for lbl in self._labels:
            with open(lbl, 'r') as j:
                jfile = json.load(j)
                img = cv2.imread(f'{self._photos}/{jfile["task"]["data"]["image"].split("/")[-1]}', 0)
                img = cv2.equalizeHist(img)
                label = [] if len(jfile["result"]) == 0 else jfile["result"][0]["value"]["choices"][0]
                self._img_label_lst.append([label, tensor(img).unsqueeze(0)])

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        label, img = self._img_label_lst[idx]
        if self._transform:
            img = self._transform(img)
        if self._target_transform:
            label = self._target_transform(label)
        return img.float(), tensor(self.classes[label])
