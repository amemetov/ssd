import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image as pil_image

from bbox_codec import BBoxCodec


class Generator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        gtb: a dictionary where key is image file name,
        value is a numpy tensor of shape (num_boxes, 4 + num_classes), num_classes without background.
    """
    def __init__(self, gtb, img_dir, bbox_codec, image_size=(300, 300)):
        self.gtb = gtb
        self.img_dir = img_dir
        self.bbox_codec = bbox_codec
        self.image_size = image_size

    def _augment_data(self, img, y):
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
                    img = load_img(img_full_path, target_size=self.image_size)
                    img = img_to_array(img)

                    # get the origin GTBs
                    y = self.gtb[img_file_name].copy()

                    # Do data augmentation
                    img, y = self._augment_data(img, y)

                    # Convert origin GTBs to format expected by NN
                    y_encoded = self.bbox_codec.encode(y)

                    x_batch.append(img)
                    y_batch.append(y_encoded)

                yield x_batch, y_batch

def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img

def img_to_array(img, data_format='channels_last'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    # if data_format is None:
    #     data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=np.float32)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


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

