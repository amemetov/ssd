import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from ssd.bbox_codec import BBoxCodec
from ssd import imaging


class Generator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        gtb: a dictionary where key is image file name,
        value is a numpy tensor of shape (num_boxes, 4 + num_classes), num_classes without background.
    """
    def __init__(self, gtb, img_registry, target_img_size, data_augmenter, bbox_codec):
        self.gtb = gtb
        self.img_registry = img_registry
        self.target_img_size = target_img_size
        self.data_augmenter = data_augmenter
        self.bbox_codec = bbox_codec

    def flow(self, img_file_names, batch_size=32, do_augment=True, do_shuffle=True, return_debug_info=False):
        num_samples = len(img_file_names)
        while True:
            if do_shuffle:
                samples = shuffle(img_file_names)
            else:
                samples = img_file_names

            for offset in range(0, num_samples, batch_size):
                samples_batch = samples[offset:offset + batch_size]

                x_batch = []
                y_batch = []

                # debug info
                orig_x_batch = []
                orig_y_batch = []
                matches_batch = []

                for img_file_name in samples_batch:
                    if img_file_name not in self.gtb:
                        #print('File {} is not in gtb'.format(img_file_name))
                        continue

                    img = self.img_registry.get(img_file_name).astype(np.float32)

                    # get the origin GTBs
                    y = self.gtb[img_file_name].copy()

                    # work with the copy of y
                    y = np.copy(y)

                    # get rid of the last 'is_difficult' column
                    y = y[:, :-1]

                    # Do data augmentation
                    if do_augment:
                        img, y = self.data_augmenter.augment(img, y)

                    # skip samples which have no GTB
                    if y.shape[0] == 0:
                        continue

                    orig_x_batch.append(img.astype(np.uint8))
                    orig_y_batch.append(np.copy(y))

                    # Preprocess image
                    img = imaging.preprocess_img(img, self.target_img_size)

                    # Convert origin GTBs to format expected by NN
                    y_encoded, assign_result = self.bbox_codec.encode(y)
                    matches_batch.append(assign_result)

                    # skip samples which have no matched PB <-> GTB (i.e. all PBs are backgrounds)
                    if len(y_encoded.shape) > 1 and np.sum(y_encoded[:, 4]) == self.bbox_codec.num_priors:
                        continue

                    x_batch.append(img)
                    y_batch.append(y_encoded)

                if return_debug_info:
                    yield np.array(x_batch), np.array(y_batch), orig_x_batch, orig_y_batch, matches_batch
                else:
                    yield np.array(x_batch), np.array(y_batch)


if __name__ == '__main__':
    import pickle
    import prior_box as pb

    num_classes = 20
    prior_boxes = pb.create_prior_boxes_vect(300, 300, pb.default_config, pb.default_prior_variance)
    bbox_codec = BBoxCodec(prior_boxes, num_classes)

    img_dir = '../datasets/voc2012/VOCtrainval_11-May-2012/JPEGImages/'
    gtb = pickle.load(open('data/pascal_voc_2012_trainval.p', 'rb'))
    gen = Generator(gtb, img_dir, bbox_codec)

    img_file_names = shuffle(list(gtb.keys()))
    train_imgs, valid_imgs = train_test_split(img_file_names, test_size=0.1)

    res = gen.flow(train_imgs)
    print(res)

    for X, y in res:
        print(y)

