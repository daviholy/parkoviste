import cv2
import numpy as np
import json
import requests
from datetime import datetime


headers = {
    "Authorization": "Bearer ##secret key##"}
img_dir = "./photos/"


def save_img_to_drive(img_name):
    global headers
    global img_dir
    para = {
        "name": img_name,
        "parents": ["1iPsJb27oM8SQFLPf_E2tMiDEiac-9Caz"]
    }
    files = {
        'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
        'file': open(img_dir+img_name, "rb")
    }
    r = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files
    )
    print(r.text)


cap = cv2.VideoCapture(0)  # fill rtsp
# example 'rtsp://username:password@192.168.1.64/1'

ret, frame = cap.read()
if not ret:
    print("No camera return")
else:
    img_name = datetime.now().strftime("%Y_%b_%d_%H_%M") + ".png"
    cv2.imwrite(img_dir+img_name, frame)
    save_img_to_drive(img_name)


cap.release()
cv2.destroyAllWindows()
