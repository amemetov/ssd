import os
import numpy as np
from xml.etree import ElementTree
import pickle
import h5py

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from ssd import imaging
#from . import imaging
#import imaging

class ImgRegistry(object):
    def get(self, img_name):
        raise NotImplementedError()

class FsImgRegistry(ImgRegistry):
    def __init__(self, img_dirs):
        self.img_dirs = img_dirs

    def get(self, img_name):
        for img_dir in self.img_dirs:
            img_full_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_full_path):
                img = imaging.load_img(img_full_path).astype(np.float32)
                return img
        raise ValueError("No {} exists in dirs {}".format(img_name, self.img_dirs))

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

def ensure_list(arg):
    if type(arg) is not list:
        arg = [arg]
    return arg

class PascalVocData(object):
    CLASSES = [
        '__background__',
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, trainval_gtb_files, test_gtb_files, trainval_samples_files, test_samples_files,
                 trainval_img_dirs, test_img_dirs, val_size=0.2):
        trainval_gtb_files = ensure_list(trainval_gtb_files)
        test_gtb_files = ensure_list(test_gtb_files)
        trainval_samples_files = ensure_list(trainval_samples_files)
        test_samples_files = ensure_list(test_samples_files)
        trainval_img_dirs = ensure_list(trainval_img_dirs)
        test_img_dirs = ensure_list(test_img_dirs)

        self.trainval_gtb = self.__load_gtbs(trainval_gtb_files)
        self.test_gtb = self.__load_gtbs(test_gtb_files)

        trainval_samples = shuffle(self.__load_samples(trainval_samples_files))
        self.train_samples, self.valid_samples = train_test_split(trainval_samples, test_size=val_size)
        self.test_samples = self.__load_samples(test_samples_files)

        # PickleImgRegistry('./voc2012-images.pickle')
        # Hdf5ImgRegistry('./voc2012-images.hdf5')
        self.trainval_img_registry = FsImgRegistry(trainval_img_dirs)
        self.test_img_registry = FsImgRegistry(test_img_dirs)

    def get_trainval_gtb(self):
        return self.trainval_gtb

    def get_test_gtb(self):
        return self.test_gtb

    def get_trainval_samples(self):
        return self.train_samples, self.valid_samples

    def get_test_samples(self):
        return self.test_samples

    def get_trainval_img_registry(self):
        return self.trainval_img_registry

    def get_test_img_registry(self):
        return self.test_img_registry

    def __load_gtbs(self, gtb_files):
        gtb = {}
        for gtb_file in gtb_files:
            with open(gtb_file, 'rb') as f:
                gtb.update(pickle.load(f))
        return gtb

    def __load_samples(self, samples_files):
        samples = []
        for samples_file in samples_files:
            samples += load_samples_list(samples_file)
        return samples



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

    def _is_valid_class(self, name):
        return name in PascalVocData.CLASSES

    def _to_one_hot(self, name):
        nb_classes = len(PascalVocData.CLASSES)
        one_hot_vector = [0] * nb_classes
        # method index throws error if there is no such item
        ind = PascalVocData.CLASSES.index(name)
        one_hot_vector[ind] = 1
        return one_hot_vector

# python ssd/data.py '../datasets/voc/VOCtest_06-Nov-2007/Annotations/' 'data/pascal_voc_2007_test.p'
# python ssd/data.py '../datasets/voc/VOCtrainval_06-Nov-2007/Annotations/' 'data/pascal_voc_2007_trainval.p'
# python ssd/data.py '../datasets/voc/VOCtrainval_11-May-2012/Annotations/' 'data/pascal_voc_2012_trainval.p'
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