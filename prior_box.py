import pickle
import numpy as np
import pdb

img_width, img_height = 300, 300


# see https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp

def create_prior_boxes(box_config):
    for layer_config in box_config:
        layer_width = layer_config['layer_width']
        layer_height = layer_config['layer_height']
        min_size = layer_config['min_size']
        max_size = layer_config['max_size']
        aspect_ratios = layer_config['aspect_ratios']
