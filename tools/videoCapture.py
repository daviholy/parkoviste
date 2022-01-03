import cv2
import argparse
from datetime import datetime

"""
Small utility, for taking photo from camera trough rtsp protocol
"""

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='./photos/',
                    help='directory where camera images should be stored')
parser.add_argument('-cs', '--camera_source', default=0,
                    help='camera source is used primarily ,example: rtsp://username:password@192.168.1.64/1')
parser.add_argument('-pre', '--prefix', type=str, default="",
                    help="photo name prefix, example: 'cam1_'...jpg")
args = parser.parse_args()

cap = cv2.VideoCapture(args.camera_source)
ret, frame = cap.read()
if not ret:
    print("No camera return")
else:
    img_name = args.prefix + datetime.now().strftime("%Y_%b_%d_%H_%M") + ".jpg"
    cv2.imwrite(args.img_dir + "/" + img_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
