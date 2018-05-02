import numpy as np
import cv2
from scipy.misc.pilutil import imread

def load_img(path):
    # loads image as uint8
    return imread(path)

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