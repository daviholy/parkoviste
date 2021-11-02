import cv2
import json
import requests
import argparse
from datetime import datetime
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument('client_credentials', type=str, help='json file with client credentials')
parser.add_argument('--img_dir', type=str, default='./photos/', help='directory where camera images should be stored')
parser.add_argument('--camera_source', default=0, help='example: rtsp://username:password@192.168.1.64/1')
args = parser.parse_args()

credentials = {}

try:
    with open(args.client_credentials, 'r') as cr:
        credentials = json.load(cr)
except FileNotFoundError:
    exit("File " + args.client_credentials + " not found")


def save_img_to_drive(img_name, binary_img):
    token = credentials['access_token']
    para = {
        "name": img_name,
        "parents": ["1iPsJb27oM8SQFLPf_E2tMiDEiac-9Caz"]}  # google drive directory id
    files = {
        'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
        'file': binary_img}

    r = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                      headers={"Authorization": "Bearer " + token}, files=files)

    if r.status_code == 401:  # token expired - unauthorized
        token = refresh_access_token()
        credentials['access_token'] = token
        with open(args.client_credentials, 'w') as fp:
            json.dump(credentials, fp)
        # Try to send picture once more with new token
        r = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                          headers={"Authorization": "Bearer " + token}, files=files)

    return r.status_code == 200


def refresh_access_token():
    # Ask server for new access token and save it to the json file
    r = requests.post(
        'https://www.googleapis.com/oauth2/v4/token',
        headers={'content-type': 'application/x-www-form-urlencoded'},
        data={
            'grant_type': 'refresh_token',
            'client_id': credentials['client_id'],
            'client_secret': credentials['client_secret'],
            'refresh_token': credentials['refresh_token'],
        }
    )
    if r.status_code == 200:
        return r.json()['access_token']


cap = cv2.VideoCapture(args.camera_source)

ret, frame = cap.read()
if not ret:
    print("No camera return")
else:
    img_name = datetime.now().strftime("%Y_%b_%d_%H_%M") + ".png"
    saved = save_img_to_drive(img_name, cv2.imencode('.png', frame)[1].tobytes())
    if not saved:
        cv2.imwrite(args.img_dir + img_name, frame)  # Backup local save if upload was unsuccessful

cap.release()
cv2.destroyAllWindows()
