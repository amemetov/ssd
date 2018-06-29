import numpy as np
import cv2
from scipy.misc.pilutil import imread, imresize

def load_img(path):
    # loads image as uint8 cause below augmentation methods expect RGB image with type uint8
    #return imread(path)

    flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    img = cv2.imread(path, flags)
    if img is None:
        raise OSError("Can't read an image: {}".format(path))
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        raise OSError("Can't convert image {} using COLOR_BGR2RGB".format(path)) from e

def resize_img(img, target_img_size):
    if img.shape[:2] == target_img_size[:2]:
        return img
    return imresize(img, target_img_size).astype('float32')

IMAGENET_RGB_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)
MEAN = IMAGENET_RGB_MEAN

def normalize_img(img, mean=MEAN):
    # Zero-center by mean pixel
    img -= mean

    # Squash to about [-0.5, 0.5]
    img = img / 255.0

    # Convert RGB -> BGR
    img = img[..., ::-1]

    return img

def denormalize_img(img, mean=MEAN):
    # Convert BGR -> RGB
    img = img[..., ::-1]

    # Undo squashing
    img = img * 255

    # Undo zero-center by mean pixel
    img += mean

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def preprocess_img(img, target_img_size):
    img = resize_img(img, target_img_size)
    img = normalize_img(img)
    return img

def flip_horiz(img):
    return np.fliplr(img)

def randomize_brightness(img, hsv=None, brightness_delta=32):
    # brightness = np.random.uniform(-brightness_delta, brightness_delta)
    # img = np.clip(img + brightness, 0, 255)
    # return img

    if hsv is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    brightness = -np.random.uniform(-brightness_delta, brightness_delta)
    # V in range [0, 255]
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), hsv


def randomize_contrast(img, hsv=None, contrast_lower=0.5, contrast_upper=1.5):
    # contrast = np.random.uniform(contrast_lower, contrast_upper)
    # img = np.clip(contrast * img, 0, 255)
    # return img

    if hsv is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    contrast = np.random.uniform(contrast_lower, contrast_upper)
    # V in range [0, 255]
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * contrast, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), hsv

def randomize_hue(img, hsv=None, hue_delta=18):
    if hsv is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = np.random.uniform(-hue_delta, hue_delta)
    # H in range [0, 360]
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue, 0, 360)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), hsv

def randomize_saturation(img, hsv=None, saturation_lower=0.5, saturation_upper=1.5):
    if hsv is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = np.random.uniform(saturation_lower, saturation_upper)
    # S in range [0, 1]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), hsv