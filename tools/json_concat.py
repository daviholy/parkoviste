import argparse
import json
import os.path
import datetime


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


def concat(json_new, json_ori, position):
    first_d = dict(list(json_ori.items())[:position])
    second_d = dict(list(json_ori.items())[position:])

    json_new_renamed = {}
    for data in json_new:
        temp = (data.split('-')[0] + '-' + str(int(data.split('-')[1]) + position))
        json_new_renamed[temp] = json_new[data]

    first_two_concat = {**first_d, **json_new_renamed}
    last_index = len(first_two_concat)
    
    counter = 0
    second_d_renamed = {}
    for data in second_d:
        temp = (data.split('-')[0] + '-' + str((last_index + counter)))
        second_d_renamed[temp] = second_d[data]
        counter += 1

    final_concat = {**first_two_concat, **second_d_renamed}

    save_to_json(final_concat)


def main():
    """
    Set up of parser and calls for script functions
    """
    parser = argparse.ArgumentParser(description='Object position coordination from rectangles in image')
    parser.add_argument('-jn', '--json_new', type=str,
                        default="parkingPlaces.json", help='Path to the json with new positions')
    parser.add_argument('-jo', '--json_original', type=str,
                        default="predictions.json", help='Path to the original json')
    parser.add_argument('-po', '--position', type=int, help='Positions where to concat')
    args = parser.parse_args()

    json_new = read_json(args.json_new)
    json_ori = read_json(args.json_original)
    concat(json_new, json_ori, args.position)


if __name__ == "__main__":
    main()
