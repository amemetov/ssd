import os
import numpy as np
from xml.etree import ElementTree
import pickle
import h5py

from . import imaging

class ImgRegistry(object):
    def get(self, img_name):
        raise NotImplementedError()

class FsImgRegistry(ImgRegistry):
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def get(self, img_name):
        img_full_path = os.path.join(self.img_dir, img_name)
        img = imaging.load_img(img_full_path).astype(np.float32)
        return img

class PickleImgRegistry(ImgRegistry):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.pickle_file = pickle.load(f)

    def get(self, img_name):
        return self.pickle_file[img_name]

class Hdf5ImgRegistry(ImgRegistry):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(self.hdf5_path, "r")

    def close(self):
        self.hdf5_file.close()

    def get(self, img_name):
        img = self.hdf5_file[img_name]
        return np.array(img)

def images_to_pickle(img_dir, pickle_path, img_files=None, target_img_size=None):
    # datasets/voc/VOCtrainval_06-Nov-2007/JPEGImages
    if not img_files:
        img_files = os.listdir(img_dir)
    print("Found '{}' files in dir '{}'".format(len(img_files), img_dir))

    with open(pickle_path, 'wb') as f:
        data = {}
        for img_file in img_files:
            img_full_path = os.path.join(img_dir, img_file)
            img = imaging.load_img(img_full_path)
            if target_img_size:
                img = imaging.resize_img(img, target_img_size)
            data[img_file] = img.astype(np.uint8)

        pickle.dump(data, f)


def images_to_hdf5(img_dir, hdf5_path, img_files=None, target_img_size=None):
    # datasets/voc/VOCtrainval_06-Nov-2007/JPEGImages
    if not img_files:
        img_files = os.listdir(img_dir)
    print("Found '{}' files in dir '{}'".format(len(img_files), img_dir))

    with h5py.File(hdf5_path, mode='w') as hdf5_file:
        for img_file in img_files:
            img_full_path = os.path.join(img_dir, img_file)
            img = imaging.load_img(img_full_path)
            if target_img_size:
                img = imaging.resize_img(img, target_img_size)
            hdf5_file[img_file] = img.astype(np.uint8)



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
        '__background__',
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