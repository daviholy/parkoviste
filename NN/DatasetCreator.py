from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import ImageReadMode
from torchvision.io import read_image
from torch import tensor
from os import path
import os
import json


class DatasetCreator(Dataset):
    def __init__(self, label_dict: dict, dest_photos="dataset/photos", dest_labels="dataset/labels", transform=None, target_transform=None):
        # print(os.getcwd())
        if not path.isdir(Path(dest_photos)):
            exit("Not a valid image directory")
        if not path.isdir(Path(dest_labels)):
            exit("Not a valid label directory")
        self.classes = label_dict
        self.labels = sorted(Path(dest_labels).glob("*.json"))
        self.transform = transform
        self.target_transform = target_transform
        self.photos = dest_photos

        self.img_label_lst = []
        for lbl in self.labels:
            with open(lbl, 'r') as j:
                jfile = json.load(j)
                img = read_image(f'{self.photos}/{jfile["task"]["data"]["image"].split("/")[-1]}', ImageReadMode.RGB)
                label = [] if len(jfile["result"]) == 0 else jfile["result"][0]["value"]["choices"][0]
                self.img_label_lst.append([label, img])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label, img = self.img_label_lst[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img.float(), tensor(self.classes[label])
