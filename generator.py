import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from bbox_codec import BBoxCodec
import imaging


class Generator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        gtb: a dictionary where key is image file name,
        value is a numpy tensor of shape (num_boxes, 4 + num_classes), num_classes without background.
    """
    def __init__(self, gtb, img_dir, data_augmenter, bbox_codec, image_size=(300, 300)):
        self.gtb = gtb
        self.img_dir = img_dir
        self.data_augmenter = data_augmenter
        self.bbox_codec = bbox_codec
        self.image_size = image_size

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
                    img = imaging.load_img(img_full_path).astype(np.float32)

                    # get the origin GTBs
                    y = self.gtb[img_file_name].copy()

                    # Do data augmentation
                    img, y = self.data_augmenter.augment(img, y)

                    # normalize
                    img = img / 127.5 - 1.0

                    # Convert origin GTBs to format expected by NN
                    y_encoded = self.bbox_codec.encode(y)

                    x_batch.append(img)
                    y_batch.append(y_encoded)

                yield np.array(x_batch), np.array(y_batch)


if __name__ == '__main__':
    import pickle
    import prior_box as pb

    num_classes = 20
    prior_boxes = pb.create_prior_boxes_vect(300, 300, pb.default_config, pb.default_prior_variance)
    bbox_codec = BBoxCodec(prior_boxes, num_classes)

    img_dir = '../datasets/voc2012/VOCtrainval_11-May-2012/JPEGImages/'
    gtb = pickle.load(open('data/pascal_voc_2012.p', 'rb'))
    gen = Generator(gtb, img_dir, bbox_codec)

    img_file_names = shuffle(list(gtb.keys()))
    train_imgs, valid_imgs = train_test_split(img_file_names, test_size=0.1)

    res = gen.flow(train_imgs)
    print(res)

    for X, y in res:
        print(y)

