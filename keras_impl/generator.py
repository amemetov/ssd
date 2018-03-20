import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import os


class Generator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        gtb: a dictionary where key is image file name,
        value is a numpy tensor of shape (num_boxes, 4 + num_classes), num_classes without background.
    """
    def __init__(self, gtb, img_dir):
        self.gtb = gtb
        self.img_dir = img_dir

    def _randomize_img(self, img, y):
        # TODO: implement
        return img, y

    def flow(self, img_file_names, batch_size=32):
        num_samples = len(img_file_names)
        while True:
            samples = shuffle(img_file_names)

            for offset in range(0, num_samples, batch_size):
                samples_batch = samples[offset:offset + batch_size]

                x_batch = []
                y_batch = []

                for img_file_name in samples_batch:
                    img_full_path = os.path.join(self.img_dir, img_file_name)
                    img = plt.imread(img_full_path)#.astype('float32')
                    y = self.gtb[img_file_name].copy()

                    img, y = self._randomize_img(img, y)

                    x_batch.append(img)
                    y_batch.append(y)

                yield x_batch, y_batch




