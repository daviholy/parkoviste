if __name__ == "__main__":
    from common import Common
    
from pathlib import Path
import json
import argparse
import numpy as np
from shutil import copy
import os

""" Data splitting script

Randomly splits labels and corresponding images to two directories - testing, training.
"""

class DataSplitter:
    @staticmethod
    def _clear_source_labels(labels):
        """
        Clears source json files with labels from skipped annotations.

        :param labels: paths to json files with labels
        :return: list of tuples - [['json_file_name', 'img_file_name'], ...]
        :rtype: list
        """
        json_img = []
        # remove json files with label and create dict - json_file_name: img_name
        for l in labels:
            jFile = None

            with open(l, 'r') as j:
                jfile = json.load(j)

            if len(jfile["result"]) == 0 or not jfile["task"]["is_labeled"] or jfile["was_cancelled"]:
                # skipped annotation, continue to another json file
                continue
            json_file_name = l.parts[-1]
            img_name = f'{jfile["task"]["data"]["image"].split("/")[-1]}'
            json_img.append([json_file_name, img_name])

        return json_img

    @staticmethod
    def _get_highest_label_id(labels_dir):
        """
        Gets highest json file id.

        :param labels_dir: directory of paths to json label files
        :return: highest json file id
        :rtype: int
        """
        labels = Path(labels_dir).glob("*.json")
        labels = sorted(labels, key=lambda l: int(l.parts[-1].split('.')[0]))
        if len(labels) == 0:
            return 0
        return int(labels[-1].parts[-1].split('.')[0])

    @classmethod
    def split_source_data(self,shuffle=True, random_seed=42):
        """
        Randomly splits labels and corresponding images to two directories (testing, training).
        Also checks if destination directories contains json files and split only new labels.

        :param shuffle: decides if photos will be randomly shuffled
        :param random_seed: serves as seed for shuffling
        :rtype: None
        """
        labels = Path(f"{args.source_dir}/labels").glob("*.json")
        labels = sorted(labels, key=lambda l: int(l.parts[-1].split('.')[0]))

        # Search for highest label id in existing datasets directories

        cut_of_id = max(self._get_highest_label_id(training_dir['labels']),
                        self._get_highest_label_id(testing_dir['labels']))

        # Cut off existing labels (already divided in past function call)
        labels = labels[cut_of_id:]

        json_img = self._clear_source_labels(labels)

        # Split data randomly
        data_size = len(json_img)
        indices = list(range(data_size))
        split = int(np.floor(args.train_split_ratio * data_size))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        test_indices, train_indices = [indices[split:], 'testing'], [indices[:split], 'training']

        for idxs, dataset_name in [test_indices, train_indices]:
            for idx in idxs:
                jfile_name, img_name = json_img[idx]
                if args.copy_mode:
                    # Copy files
                    copy(f'{args.source_dir}/photos/{img_name}', dest_dirs[dataset_name]["photos"])
                    copy(f'{args.source_dir}/labels/{jfile_name}', dest_dirs[dataset_name]["labels"])
                else:
                    # Replace files
                    os.replace(f'{args.source_dir}/labels/{jfile_name}',f'{dest_dirs[dataset_name]["labels"]}/{jfile_name}')
                    os.replace(f'{args.source_dir}/photos/{img_name}',f'{dest_dirs[dataset_name]["photos"]}/{img_name}')

    @classmethod
    @Common.debug("duplicated images in dataset")
    def check_for_duplicate_annotations(self):
        """
        Debug function that checks for multiple annotations for one photo.

        :return:
        """
        data = self.clear_source_labels(Path(f"{args.source_dir}/labels").glob("*.json"))
        counter = 0
        for j, i in data:
            for jj, ii in data:
                if j != jj and i == ii:
                    counter += 1
                    print(f'json files\n{j}\n{jj}\nand images\n{i}\n{ii}')
                    print('--------------------------------------------------------')

        print(counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--source_dir', type=str, default='../dataset',
                        help='directory where the source dataset is stored (exported from label studio)')
    parser.add_argument('-dd', '--destination_dir', type=str, default='../dataset/split',
                        help='directory where the split datasets should be stored')
    parser.add_argument('-sr', '--train_split_ratio', type=float, default=0.7,
                        help='float in range 0-1 that determines ratio of slit between train and test dataset')
    parser.add_argument('-cm', '--copy_mode', type=bool, default=False,
                        help='decides if data will be copied od replaced, copy is slow, replace is faster')
    args = Common.common_arguments(parser)

    training_dir = {'labels': f'{args.destination_dir}/training/labels',
                    'photos': f'{args.destination_dir}/training/photos'}
    testing_dir = {'labels': f'{args.destination_dir}/testing/labels',
                   'photos': f'{args.destination_dir}/testing/photos'}

    dest_dirs = {'training': training_dir, 'testing': testing_dir}

    DataSplitter.split_source_data()
    # DataSplitter.check_for_duplicate_annotations()
