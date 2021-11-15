import argparse
from cv2 import cv2
import json

drawing = False
point1 = ()
point2 = ()
ar = []
type_p = ""


def mouse_drawing(event, x, y, flags, params):
    global point1, point2, drawing, ar, type_p

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            point1 = (x, y)
            drawing = True
        else:
            drawing = False
            ar.append([point1, point2, type_p])
            point1 = ()
            point2 = ()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            point2 = (x, y)


def save_to_json(data):
    with open('coordinates.json', 'w') as file:
        json.dump(data, file)

def read_json():
    data = {}
    with open('coordinates.json', 'r') as file:
        data = json.load(file)
    return data


def legend(image):
    cv2.putText(
        image,  # numpy array on which text is written
        "S - standard | D - disabled | N - non-standard",  # text
        (10, 15),  # position at which writing has to start
        cv2.FONT_HERSHEY_DUPLEX,  # font family
        0.3,  # font size
        (255, 255, 255, 255),  # font color
        1) #font stroke
    cv2.putText(
        image,  # numpy array on which text is written
        "Z - undo | ESC - save and exit",  # text
        (10, 30),  # position at which writing has to start
        cv2.FONT_HERSHEY_DUPLEX,  # font family
        0.3,  # font size
        (255, 255, 255, 255),  # font color
        1)  # font stroke

def rectangle_opencv(image, standard_p, non_standard_p, disabled_p, color):
    global type_p

    while True:
        image_copy = image.copy()

        legend(image_copy)

        key = cv2.waitKey(1)  # wait 1ms for key otherwise continue
        if key == 115:  # 115 key code for s
            standard_p = True
            non_standard_p = False
            disabled_p = False
            color = (0, 255, 0)

        if key == 110:  # 110 key code for n
            standard_p = False
            non_standard_p = True
            disabled_p = False
            color = (255, 0, 0)  # BGR

        if key == 100:  # 100 key code for d
            standard_p = False
            non_standard_p = False
            disabled_p = True
            color = (0, 0, 255)

        if standard_p:
            type_p = "standard"
        elif non_standard_p:
            type_p = "non-standard"
        elif disabled_p:
            type_p = "disabled"

        if point1 and point2:
            cv2.rectangle(image_copy, point1, point2, color)

        if ar:
            for i in range(len(ar)):
                if ar[i][2] == "standard":
                    cv2.rectangle(image_copy, ar[i][0], ar[i][1], (0, 255, 0))
                elif ar[i][2] == "non-standard":
                    cv2.rectangle(image_copy, ar[i][0], ar[i][1], (255, 0, 0))
                elif ar[i][2] == "disabled":
                    cv2.rectangle(image_copy, ar[i][0], ar[i][1], (0, 0, 255))

        cv2.imshow("Window", image_copy)

        if key == 122:  # 122 key code for z
            if ar:
                ar.pop()

        if key == 27:  # 27 key code for esc
            counter = 0
            data = {}
            data_temp = {}
            for i in ar:
                data_temp["type"] = i[2]
                data_temp["coordinates"] = [i[0], i[1]]
                data['position-' + str(counter)] = data_temp
                counter += 1
                data_temp = {}

            save_to_json(data)
            break

    cv2.destroyAllWindows()

def view(image):
    data = read_json()
    type_ar = []
    coor_ar = []

    if data:
        for i in range(len(data)):
            type_ar.append(data['position-' + str(i)]["type"])
            coor_ar.append(data['position-' + str(i)]["coordinates"])
        for i in range(len(type_ar)):
            if type_ar[i] == "standard":
                cv2.rectangle(image, coor_ar[i][0], coor_ar[i][1], (0, 255, 0))
            elif type_ar[i] == "non-standard":
                cv2.rectangle(image, coor_ar[i][0], coor_ar[i][1], (255, 0, 0))
            elif type_ar[i] == "disabled":
                cv2.rectangle(image, coor_ar[i][0], coor_ar[i][1], (0, 0, 255))

        cv2.putText(
            image,  # numpy array on which text is written
            "View",  # text
            (10, 15),  # position at which writing has to start
            cv2.FONT_HERSHEY_DUPLEX,  # font family
            0.5,  # font size
            (0, 150, 255, 255),  # font color
            1)  # font stroke

        cv2.imshow("Window", image)
        cv2.waitKey(0)
    else:
        print("No data to display.")


def main():
    parser = argparse.ArgumentParser(description='Object position coordination from rectangles in image')
    parser.add_argument('imagePath', type=str, help='Path to image.')
    parser.add_argument('--view', type=bool, default=False, help='False - default value | True - rectangle viewer')
    args = parser.parse_args()

    cv2.namedWindow("Window")
    cv2.setMouseCallback("Window", mouse_drawing)
    image = cv2.imread(args.imagePath)

    if not args.view:
        rectangle_opencv(image, standard_p=True, non_standard_p=False, disabled_p=False,
                         color=(0, 255, 0))
    else:
        view(image)

if __name__ == "__main__":
    main()
