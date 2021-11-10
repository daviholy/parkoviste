import argparse
from cv2 import cv2
import json

drawing = False
point1 = ()
point2 = ()
ar = []

def mouse_drawing(event, x, y, flags, params):
    global point1, point2, drawing, ar

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            point1 = (x, y)
            drawing = True
        else:
            drawing = False
            ar.append([point1, point2])
            point1 = ()
            point2 = ()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            point2 = (x, y)


def save_to_json(data):
    with open('coordinates.json', 'w') as file:
        json.dump(data, file)


def main():
    parser = argparse.ArgumentParser(description='Object position coordination from rectangles in image')
    parser.add_argument('imagePath', type=str, help='Path to image.')
    args = parser.parse_args()

    cv2.namedWindow("Window")
    cv2.setMouseCallback("Window", mouse_drawing)

    while True:
        image = cv2.imread(args.imagePath)

        if point1 and point2:
            cv2.rectangle(image, point1, point2, (0, 255, 0))

        if ar:
            for i in range(len(ar)):
                cv2.rectangle(image, ar[i][0], ar[i][1], (0, 255, 0))

        cv2.imshow("Window", image)

        key = cv2.waitKey(1)  # wait 1ms for key otherwise continue

        if key == 122:  # 122 key code for z
            if ar:
                ar.pop()

        if key == 27:  # 27 key code for esc
            counter = 0
            data = {}
            for i in ar:
                data['rectangle-' + str(counter)] = i
                counter += 1
            save_to_json(data)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()