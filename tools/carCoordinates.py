import argparse
import cv2
import json
import os.path
import datetime

"""
Script that enable user to create rectangles on picture of type standard - green,
non-standard - blue, and disabled - red. If user press escape, rectangle coordinates 
are saved into json file.
Rectangles can be loaded from json file to be showed on the picture.
"""

drawing = False
is_edited = False
is_adding = False
add_index = -1
point1 = ()
point2 = ()
mouse_position = ()
ar = []
type_p = "standard"
index = 0


def mouse_drawing(event, x, y, flags, params):
    """
    :param event:
    :param x: X coordinates of mouse
    :param y: Y coordinates of mouse
    :param flags:
    :param params:
    """
    global point1, point2, drawing, ar, type_p, mouse_position, is_edited, index, is_adding, add_index

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            point1 = (x, y)
            drawing = True
        else:
            drawing = False
            if point1[0] >= point2[0] and point1[1] >= point2[1]:
                temp = point1
                point1 = point2
                point2 = temp
            elif point1[0] >= point2[0]:
                temp = point1
                point1 = (point2[0], point1[1])
                point2 = (temp[0], point2[1])
            elif point1[1] >= point2[1]:
                temp = point1
                point1 = (point1[0], point2[1])
                point2 = (point2[0], temp[1])

            if is_edited:
                ar[index] = ([point1, point2, type_p])
                is_edited = False
            if is_adding:
                ar.insert(add_index, [point1, point2, type_p])
                is_adding = False
            if not is_edited and not is_adding:
                ar.append([point1, point2, type_p])

            point1 = ()
            point2 = ()
    elif event == cv2.EVENT_MOUSEMOVE:
        mouse_position = (x, y)
        if drawing:
            point2 = (x, y)


def save_to_json(data):
    """
    :param data: Data to be saved to file
    """
    date = datetime.datetime.now()
    date_format = date.strftime("%Y_%d_%m_%H_%M_%S")
    with open(f"parkingPlace{date_format}.json", 'w') as file:
        json.dump(data, file)


def read_json(path):
    """
    :param path: Path to json file
    """
    with open(path, 'r') as file:
        data = {}
        try:
            data = json.load(file)
        except json.decoder.JSONDecodeError:
            print('Empty File Error!')
            return None
    return data


def legend(image):
    """
    Creates text legend in the picture for keyboard keys that change behaviour of application
    :param image: OpenCV image
    """
    cv2.putText(
        image,  # numpy array on which text is written
        "S - standard | D - disabled | N - non-standard",  # text
        (10, 15),  # position at which writing has to start
        cv2.FONT_HERSHEY_DUPLEX,  # font family
        0.5,  # font size
        (255, 255, 255, 255),  # font color
        1)  # font stroke
    cv2.putText(image, "Backspace - edit rectangle on mouse position", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                (255, 255, 255, 255), 1)
    cv2.putText(image, "Z - undo | ESC - save and exit", (10, 45), cv2.FONT_HERSHEY_DUPLEX, 0.5,
        (255, 255, 255, 255), 1)


def rectangle_opencv(image, path, edit):
    """
     Enable user to create rectangles on image.
     Green for standard position, red for disabled and blue for non-standard.
     Show all rectangles that has been created already.
     Include edit mode - load all rectangles from json file
    :param image: OpenCV image
    :param path: Path to json file
    """
    global type_p, is_edited, index, add_index, is_adding
    color_arr = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (186, 21, 228)]
    color = color_arr[0]  # Default value
    image_copy = image.copy()

    legend(image)  # Show legend for keys

    # Edit mode enabled
    if os.path.isfile(path) and edit:
        data_json = read_json(path)
        type_ar = []
        coor_ar = []
        if data_json:
            counter = 0
            for i in data_json:
                type_ar.append(data_json[i]["type"])
                coor_ar.append(data_json[i]["coordinates"])
                ar.append([coor_ar[counter][0], coor_ar[counter][1], type_ar[counter]])
                counter += 1

            for typ, coor in zip(type_ar, coor_ar):
                if typ == "standard":
                    cv2.rectangle(image_copy, coor[0], coor[1], color_arr[0])
                elif typ == "non-standard":
                    cv2.rectangle(image_copy, coor[0], coor[1], color_arr[1])
                elif typ == "disabled":
                    cv2.rectangle(image_copy, coor[0], coor[1], color_arr[2])
        else:
            print("No file to show!")

    while True:
        image_copy = image.copy()

        if point1 and point2:
            cv2.rectangle(image_copy, point1, point2, color)

        if ar:
            if is_edited:
                cv2.rectangle(image_copy, ar[index][0], ar[index][1], color_arr[3])
            for i in range(len(ar)):
                if i == index and is_edited:
                    continue
                elif ar[i][2] == "standard":
                    cv2.rectangle(image_copy, ar[i][0], ar[i][1], color_arr[0])
                elif ar[i][2] == "non-standard":
                    cv2.rectangle(image_copy, ar[i][0], ar[i][1], color_arr[1])
                elif ar[i][2] == "disabled":
                    cv2.rectangle(image_copy, ar[i][0], ar[i][1], color_arr[2])

        key = cv2.waitKey(1)  # wait 1ms for key otherwise continue

        if key == 115:  # 115 key code for s
            type_p = "standard"
            color = color_arr[0]

        if key == 110:  # 110 key code for n
            type_p = "non-standard"
            color = color_arr[1]

        if key == 100:  # 100 key code for d
            type_p = "disabled"
            color = color_arr[2]

        if key == 8:  # 8 key code for backspace, edit mode
            if ar and not is_edited:
                index = 0
                for rect in ar:
                    if (mouse_position[0] >= rect[0][0] and mouse_position[0] <= rect[1][0] and
                        mouse_position[1] >= rect[0][1] and mouse_position[1] <= rect[1][1]):
                        is_edited = True
                        break
                    index += 1

        if key == 97 and not is_adding:  # 97 key code for a, adding mode
            is_adding = True
            print('Write index of added rectangle')
            add_index = int(input())
            print(f'Index was: {add_index}')

        if key == 122:  # 122 key code for z
            if ar:
                ar.pop()

        if cv2.getWindowProperty('Window', cv2.WND_PROP_VISIBLE) < 1 or key == 27: # if window has been closed app will be terminated. 27 key code for esc
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

        cv2.imshow("Window", image_copy)

    cv2.destroyAllWindows()


def view(image, path_json_positions, path_json_predictions, car_line):
    """
    Show rectangles and creates rectangles by coordinates from json file.
    :param image: OpenCV image where rectangles will be created
    :param path: Path to json file
    """
    data_json = read_json(path_json_positions)
    predicted_json = read_json(path_json_predictions)
    type_ar = []
    coor_ar = []
    counter = 0
    car_count = 0

    if data_json and predicted_json:
        for i in data_json:
            type_ar.append(data_json[i]["type"])
            coor_ar.append(data_json[i]["coordinates"])
        for typ, coor in zip(type_ar, coor_ar):
            pos = coor[0] if counter < car_line else [coor[0][0], coor[1][1]+19]
            car_count += 1 if predicted_json[f"position-{counter}"] == "1" else 0
            cv2.putText(img=image, text=f'{counter}:{predicted_json[f"position-{counter}"]}', org=pos,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
            counter += 1
            if typ == "standard":
                cv2.rectangle(image, coor[0], coor[1], (0, 255, 0))
            elif typ == "non-standard":
                cv2.rectangle(image, coor[0], coor[1], (255, 0, 0))
            elif typ == "disabled":
                cv2.rectangle(image, coor[0], coor[1], (0, 0, 255))

        cv2.putText(image, "View", (10, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 150, 255, 255), 1)
        print(f"Number of cars: {car_count}")
        cv2.imshow("Window", image)
        while cv2.getWindowProperty('Window', cv2.WND_PROP_VISIBLE) > 0:
            key = cv2.waitKey(1)
            if key == 27:
                break
    else:
        print("No data to display.")


def main():
    """
    Set up of parser and calls for script functions
    """
    parser = argparse.ArgumentParser(description='Object position coordination from rectangles in image')
    parser.add_argument('imagePath', type=str, help='Path to image.')
    parser.add_argument('-jpo', '--json_positions', type=str,
                        default="parkingPlaces.json", help='Path to the json positions file')
    parser.add_argument('-jpr', '--json_predictions', type=str,
                        default="predictions.json", help='Path to the json prediction file')
    parser.add_argument('-e', '--edit', type=bool, default=False, help='Enable user to edit rectangles from existing json file.'
                                                                        'False - Edit mode off | True - Edit mode on')
    parser.add_argument('--view', type=bool, default=False,
                        help='Viewer mod for rectangles in picture imported from Json.'
                             'False - default value | True - rectangle viewer')
    parser.add_argument('-cr', '--car_line', type=int, default=32,
                        help='Number of position where text will be put to bottom of rectangle.')
    args = parser.parse_args()

    cv2.namedWindow("Window")
    cv2.setMouseCallback("Window", mouse_drawing)
    image = cv2.imread(args.imagePath)

    if args.view:
        view(image, args.json_positions, args.json_predictions, args.car_line)
    else:
        rectangle_opencv(image, args.json_positions, args.edit)


if __name__ == "__main__":
    main()
