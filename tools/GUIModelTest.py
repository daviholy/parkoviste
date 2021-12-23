import torch
import argparse
import sys, os
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import transforms
from torch import zeros
sys.path.append('../NN')
from NeuralNetwork import NeuralNetwork
from DatasetCreator import DatasetCreator
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')
        self.setScaledContents(True)

    def setPixmap(self, image):
        super().setPixmap(image)


class AppDemo(QWidget):
    def __init__(self, model):
        super().__init__()
        self.resize(400, 400)
        self.setAcceptDrops(True)

        mainLayout = QVBoxLayout()

        self.photoViewer = ImageLabel()
        mainLayout.addWidget(self.photoViewer)

        self.setLayout(mainLayout)

        self.model = model

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)

            img = read_image(file_path, ImageReadMode.RGB)
            transform = torch.nn.Sequential(transforms.Grayscale(), transforms.RandomEqualize(p=1))
            img = transform(img).float()
            ten = zeros(1, img.shape[0], img.shape[1], img.shape[2])
            ten[0] = img
            print(self.model(ten))

            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.photoViewer.setPixmap(QPixmap(file_path))


if __name__ == "__main__":
    model_path = "../../model/model_b96_e24_lr0-0025_v0-5.pth"

    classes = ('empty', 'car')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = NeuralNetwork(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    app = QApplication(sys.argv)
    demo = AppDemo(model)
    demo.show()
    sys.exit(app.exec_())

