import cv2
import os
import json
import argparse
from sys import exit
from datetime import datetime


""" Parking places photos collecting script

This script connects to given camera source and take a picture. Picture is than
cut up to pieces (parking places) which are saved separately in jpg format.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='./photos',
                    help='directory where camera images should be stored')
parser.add_argument('-pp', '--parking_places', type=str, default='parkingPlaces.json',
                    help='json file with parking places locations')
parser.add_argument('-cs', '--camera_source', default=0,
                    help='camera source is used primarily ,example: rtsp://username:password@192.168.1.64/1')
parser.add_argument('-img', '--src_img', type=str, default='./image.png',
                    help='path to image that is used in case of no camera source')
parser.add_argument('-suf', '--suffix', type=str, default="",
                    help="photo name suffix, example: ...'_cam1'.jpg")
parser.add_argument('-pre', '--prefix', type=str, default="",
                    help="photo name prefix, example: 'cam1_'...jpg")


def load_json(file_path):
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


if __name__ == '__main__':
    args = parser.parse_args()

    # Check given image directory and add / to the end if missing
    try:
        if args.img_dir[-1] != "/":
            args.img_dir = args.img_dir + "/"
    except Exception as e:
        exit("Invalid image dictionary argument (input:'" +
             args.img_dir + "') with exception: " + str(e))

    if not os.path.isdir(args.img_dir):
        exit(f"{args.img_dir} is invalid or non existing directory")

    # Take a picture from camera or use default image
    cap = cv2.VideoCapture(args.camera_source)
    ret, frame = cap.read() if args.camera_source != 0 else (
        os.path.isfile(args.src_img), cv2.imread(args.src_img))
    if not ret:
        exit("Invalid camera/image source")
    cap.release()

    # Load parking places locations from json
    parking_places = load_json(args.parking_places)

    # Cut parking places and save them separately (in jpg 85% compression).
    i = 0
    for place, data in parking_places.items():
        x1, y1 = data['coordinates'][0]
        x2, y2 = data['coordinates'][1]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        i += 1
        img_name = args.prefix + "pl-" + str(i) + "-" + data['type'][0] + "_" + \
            datetime.now().strftime("%y-%b-%d_%H-%M") + args.suffix + ".jpg"
        cv2.imwrite(args.img_dir + img_name,
                    frame[y1:y2, x1:x2], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
