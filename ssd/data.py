import os
import numpy as np
from xml.etree import ElementTree

# see in voc_base_dir 'ImageSets/Main/train.txt', 'ImageSets/Main/val.txt', 'ImageSets/Main/trainval.txt'
def load_samples_list(fileName):
    with open(fileName, "r") as text_file:
        lines = text_file.readlines()
        result = [x.strip() + '.jpg' for x in lines]
        return result

# see http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data
class PascalVoc2012(object):
    def __init__(self, annotations_dir):
        self.annotations_dir = annotations_dir

    def load_data(self):
        data = dict()
        xml_files = os.listdir(self.annotations_dir)

        print("Found '{}' xml files in dir '{}'".format(len(xml_files), self.annotations_dir))

        for xml_file in xml_files:
            difficulties = []
            bnd_boxes = []
            one_hot_classes = []

            tree = ElementTree.parse(self.annotations_dir + xml_file)
            root = tree.getroot()
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for obj_tree in root.findall('object'):
                obj_name = obj_tree.find('name').text
                is_difficult = int(obj_tree.find('difficult').text)

                if self._is_valid_class(obj_name):
                    bb_tree = obj_tree.find('bndbox')
                    # calc positions relative to the size of the image
                    xmin = float(bb_tree.find('xmin').text) / width
                    ymin = float(bb_tree.find('ymin').text) / height
                    xmax = float(bb_tree.find('xmax').text) / width
                    ymax = float(bb_tree.find('ymax').text) / height

                    if xmax > xmin and ymax > ymin:
                        difficulties.append([is_difficult])
                        bnd_boxes.append([xmin, ymin, xmax, ymax])
                        one_hot_class = self._to_one_hot(obj_name)
                        one_hot_classes.append(one_hot_class)


            file_name = root.find('filename').text
            if len(bnd_boxes) > 0:
                difficulties = np.asarray(difficulties)
                bnd_boxes = np.asarray(bnd_boxes)
                one_hot_classes = np.asarray(one_hot_classes)
                image_data = np.hstack((bnd_boxes, one_hot_classes, difficulties))
                data[file_name] = image_data
        return data

    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def _is_valid_class(self, name):
        return name in self.CLASSES

    def _to_one_hot(self, name):
        nb_classes = len(self.CLASSES)
        one_hot_vector = [0] * nb_classes
        # method index throws error if there is no such item
        ind = self.CLASSES.index(name)
        one_hot_vector[ind] = 1
        return one_hot_vector

# python ssd/data.py '../datasets/voc/VOCtrainval_06-Nov-2007/Annotations/' 'data/pascal_voc_2007.p'
# python ssd/data.py '../datasets/voc/VOCtrainval_11-May-2012/Annotations/' 'data/pascal_voc_2012.p'
if __name__ == '__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser("Data Preparing")
    parser.add_argument('annotations_dir', help='Annotations directory')
    parser.add_argument('result_file', help='Result pickle file path')
    args = parser.parse_args()

    data = PascalVoc2012(args.annotations_dir).load_data()
    pickle.dump(data, open(args.result_file, 'wb'))

    print("Result pickle file is stored in '{}'".format(args.result_file))

