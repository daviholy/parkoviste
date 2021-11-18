import argparse
from cv2 import cv2
import json

drawing = False
point1 = ()
point2 = ()
ar = []
type_p = "standard"


def mouse_drawing(event, x, y, flags, params):
    """
    :param event:
    :param x: X coordinates of mouse
    :param y: Y coordinates of mouse
    :param flags:
    :param params:
    """
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


def save_to_json(data, path):
    """
    :param data: Data to be saved to file
    :param path: Path to json file
    :return:
    """
    with open(path, 'w') as file:
        json.dump(data, file)


def read_json(path):
    """
    :param path: Path to json file
    :return:
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def legend(image):
    """
    Creates text legend in the picture for keyboard keys that change behaviour of application
    :param image: OpenCV image
    :return:
    """
    cv2.putText(
        image,  # numpy array on which text is written
        "S - standard | D - disabled | N - non-standard",  # text
        (10, 15),  # position at which writing has to start
        cv2.FONT_HERSHEY_DUPLEX,  # font family
        0.5,  # font size
        (255, 255, 255, 255),  # font color
        1)  # font stroke
    cv2.putText(
        image,  # numpy array on which text is written
        "Z - undo | ESC - save and exit",  # text
        (10, 30),  # position at which writing has to start
        cv2.FONT_HERSHEY_DUPLEX,  # font family
        0.5,  # font size
        (255, 255, 255, 255),  # font color
        1)  # font stroke


def rectangle_opencv(image, path):
    """
     Enable user to create rectangles on image.
     Green for standard position, red for disabled and blue for non-standard.
     Show all rectangles that has been created already.
    :param image: OpenCV image
    :param path: Path to json file
    :return:
    """
    global type_p
    color_arr = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    color = color_arr[0]  # Default value
    while True:
        image_copy = image.copy()

        legend(image_copy)

        key = cv2.waitKey(1)  # wait 1ms for key otherwise continue
        if key == 115:  # 115 key code for s
            type_p = "standard"
            color = color_arr[0]

        if key == 110:  # 110 key code for n
            type_p = "non-standard"
            color = color_arr[1]  # BGR

        if key == 100:  # 100 key code for d
            type_p = "disabled"
            color = color_arr[2]

        if point1 and point2:
            cv2.rectangle(image_copy, point1, point2, color)

        if ar:
            for i in range(len(ar)):
                if ar[i][2] == "standard":
                    cv2.rectangle(image_copy, ar[i][0], ar[i][1], color_arr[0])
                elif ar[i][2] == "non-standard":
                    cv2.rectangle(image_copy, ar[i][0], ar[i][1], color_arr[1])
                elif ar[i][2] == "disabled":
                    cv2.rectangle(image_copy, ar[i][0], ar[i][1], color_arr[2])

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

            check_position_of_coordinates(data)
            save_to_json(data, path)
            break

    cv2.destroyAllWindows()


def check_position_of_coordinates(data):
    """
    Change coordinates if rectangle was not drawn from left upper corner
    :param data: Rectangle type and coordinates
    :return:
    """
    coor_ar = []
    for d in data:
        coor_ar.append(data[d]["coordinates"])

    for coor in coor_ar:
        print(coor)
        tup_1 = coor[0]
        tup_2 = coor[1]
        if tup_1[0] >= tup_2[0] and tup_1[1] >= tup_2[1]:
            coor[0] = tup_2
            coor[1] = tup_1
        elif tup_1[0] >= tup_2[0]:
            coor[0] = (tup_2[0], tup_1[1])
            coor[1] = (tup_1[0], tup_2[1])
        elif tup_1[1] >= tup_2[1]:
            coor[0] = (tup_1[0], tup_2[1])
            coor[1] = (tup_2[0], tup_1[1])


def view(image, path):
    """
    Show rectangles and creates rectangles by coordinates from json file.
    :param image: OpenCV image where rectangles will be created
    :param path: Path to json file
    :return:
    """
    data_json = read_json(path)
    type_ar = []
    coor_ar = []

    if data_json:
        for i in data_json:
            type_ar.append(data_json[i]["type"])
            coor_ar.append(data_json[i]["coordinates"])
        for typ, coor in zip(type_ar, coor_ar):
            if typ == "standard":
                cv2.rectangle(image, coor[0], coor[1], (0, 255, 0))
            elif typ == "non-standard":
                cv2.rectangle(image, coor[0], coor[1], (255, 0, 0))
            elif typ == "disabled":
                cv2.rectangle(image, coor[0], coor[1], (0, 0, 255))

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
    parser.add_argument('-j', '--json', type=str, default="parkingPlace.json", help='Path to the json file')
    parser.add_argument('--view', type=bool, default=False,
                        help='Viewer mod for rectangles in picture imported from Json.'
                             'False - default value | True - rectangle viewer')
    args = parser.parse_args()

    cv2.namedWindow("Window")
    cv2.setMouseCallback("Window", mouse_drawing)
    image = cv2.imread(args.imagePath)

    if args.view:
        view(image, args.json)
    else:
        rectangle_opencv(image, args.json)


if __name__ == "__main__":
    main()
