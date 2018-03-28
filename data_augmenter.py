import numpy as np
from scipy.misc.pilutil import imresize

import imaging

class DataAugmenter(object):
    def __init__(self, target_image_size):
        self.target_image_size = target_image_size

    def augment(self, img, y):
        # 1. Randomly Sample
        img, y = self.__randomly_sample(img, y)

        # 2. Resize to fixed size
        img = imresize(img, self.target_image_size).astype('float32')

        # 3. Horizontally flip
        img, y = self.__horizontally_flip(img, y)

        # 4. Photo-metric distortions
        img = self.__apply_photo_metric_distortions(img)

        return img, y

    def __randomly_sample(self, img, y):
        if self.__flip_coin():
            return self.__sample_patch(img, y)

        if self.__flip_coin():
            return self.__randomly_sample_patch(img, y)

        # use the entire original input image
        return img, y


    def __sample_patch(self, img, y):
        # sample a patch so that the minimum jaccard overlap
        # with the objects is 0.1, 0.3, 0.5, 0.7 or 0.9

        # TODO: Implement
        return img, y

    def __randomly_sample_patch(self, img, y):
        # Randomly sample a patch

        # TODO: Implement
        return img, y

    def __horizontally_flip(self, img, y):
        if self.__flip_coin():
            # TODO: Implement
            return img, y
        return img, y

    def __apply_photo_metric_distortions(self, img):
        if self.__flip_coin():
            img = imaging.randomize_brightness(img)

        if self.__flip_coin():
            img = imaging.randomize_contrast(img)

        if self.__flip_coin():
            img = imaging.randomize_hue(img)

        if self.__flip_coin():
            img = imaging.randomize_saturation(img)

        return img

    def __flip_coin(self):
        return True if np.random.random() < 0.5 else False