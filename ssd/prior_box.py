import math
import numpy as np

# https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_pascal.py
# https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp

# VGG16 config is taken from the origin caffe implementation
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# conv9_2 ==> 1 x 1
vgg16_mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
vgg16_layers_size=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
vgg16__aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
vgg16__steps = [8, 16, 32, 64, 100, 300]

# L2 normalize conv4_3.
vgg16_normalizations = [20, -1, -1, -1, -1, -1]

# variance used to encode/decode prior bboxes.
default_prior_variance = [0.1, 0.1, 0.2, 0.2]

def gen_prior_boxes_config(img_w, img_h, layers, normalizations, layers_size, aspect_ratios,
                           flip=True, min_ratio=20, max_ratio = 90):
    assert len(layers_size) == len(aspect_ratios)
    assert min_ratio < max_ratio

    img_min_dim = min(img_w, img_h)
    nb_layers = len(layers_size)
    ratio_step = int(math.floor((max_ratio - min_ratio) / (nb_layers - 2)))

    box_config = []
    # the first item is the special case in the origin caffe implementation
    min_size = img_min_dim * 10 / 100.
    max_size = img_min_dim * 20 / 100.
    layer_name = layers[0]
    norm = normalizations[0]
    layer_w, layer_h = layers_size[0]
    box_config.append(_create_config_item(layer_name, norm, layer_w, layer_h, min_size, max_size, aspect_ratios[0], flip))
    ratio = min_ratio
    for layer_name, norm, layer_size, ar in zip(layers[1:], normalizations[1:], layers_size[1:], aspect_ratios[1:]):
        layer_w, layer_h = layer_size
        min_size = img_min_dim * ratio / 100.
        max_size = img_min_dim * (ratio + ratio_step) / 100.
        box_config.append(_create_config_item(layer_name, norm, layer_w, layer_h, min_size, max_size, ar, flip))
        ratio += ratio_step

    return box_config

def _create_config_item(layer_name, norm, layer_w, layer_h, min_size, max_size, aspect_ratios, flip):
    first_and_second_ars = [1., 1.]
    if flip:
        aspect_ratios = first_and_second_ars + [f(x) for x in aspect_ratios for f in (lambda x: x, lambda x: 1. / x)]
    else:
        aspect_ratios = first_and_second_ars + aspect_ratios

    return {'layer': layer_name, 'normalization': norm,
            'layer_width': layer_w, 'layer_height': layer_h,
            'num_prior': len(aspect_ratios),
            'min_size': min_size, 'max_size': max_size,
            'aspect_ratios': aspect_ratios}


def create_prior_boxes_iter(img_w, img_h, box_config, variance, clip=False, offset=0.5):
    result = []
    for layer_config in box_config:
        layer_w = layer_config['layer_width']
        layer_h = layer_config['layer_height']
        min_size = layer_config['min_size']
        max_size = layer_config['max_size']
        aspect_ratios = layer_config['aspect_ratios']

        num_priors = len(aspect_ratios)

        step_w = img_w / layer_w
        step_h = img_h / layer_h

        num_boxes = num_priors * layer_w * layer_h

        num_variances = len(variance)

        # prepare box widths and heights
        box_widths, box_heights = _create_box_sizes(min_size, max_size, aspect_ratios)

        # (xmin, ymin, xmax, ymax, variance0, variance1, variance2, variance3, ...)
        prior_boxes = np.zeros((num_boxes, 4 + num_variances))

        box_idx = 0
        for h in range(0, layer_h):
            for w in range(0, layer_w):
                center_x = (w + offset) * step_w
                center_y = (h + offset) * step_h

                for box_w, box_h in zip(box_widths, box_heights):
                    # xmin
                    prior_boxes[box_idx, 0] = (center_x - box_w / 2.) / img_w
                    # ymin
                    prior_boxes[box_idx, 1] = (center_y - box_h / 2.) / img_h
                    # xmax
                    prior_boxes[box_idx, 2] = (center_x + box_w / 2.) / img_w
                    # ymax
                    prior_boxes[box_idx, 3] = (center_y + box_h / 2.) / img_h
                    box_idx += 1

        # print('box_idx = {0}'.format(box_idx))

        # clip the prior's coordinate such that it is within [0, 1]
        if clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.), 1.)

        # set the variance.
        prior_boxes[:, 4:] = variance

        result.append(prior_boxes)

    result = np.concatenate(result, axis=0)
    return result


def create_prior_boxes_vect(img_w, img_h, box_config, variance, clip=False, offset=0.5):
    result = []
    for layer_config in box_config:
        layer_w = layer_config['layer_width']
        layer_h = layer_config['layer_height']
        min_size = layer_config['min_size']
        max_size = layer_config['max_size']
        aspect_ratios = layer_config['aspect_ratios']

        prior_boxes = create_prior_boxes(img_w, img_h, layer_w, layer_h,
                                         min_size, max_size, aspect_ratios,
                                         variance, clip, offset)

        result.append(prior_boxes)

    result = np.concatenate(result, axis=0)
    return result

def create_prior_boxes(img_w, img_h, layer_w, layer_h, min_size, max_size, aspect_ratios,
                       variance, clip, offset):
    num_priors = len(aspect_ratios)

    step_w = img_w / layer_w
    step_h = img_h / layer_h

    num_boxes = num_priors * layer_w * layer_h

    num_variances = len(variance)

    # prepare box widths and heights
    box_widths, box_heights = _create_box_sizes(min_size, max_size, aspect_ratios)
    box_widths /= 2. * img_w
    box_heights /= 2. * img_h

    # (xmin, ymin, xmax, ymax, variance0, variance1, variance2, variance3, ...)
    prior_boxes = np.zeros((num_boxes, 4 + num_variances))

    widths = np.arange(0, layer_w)
    heights = np.arange(0, layer_h)
    center_x = ((widths + offset) * step_w) / img_w
    center_y = ((heights + offset) * step_h) / img_h
    # get all combinations of center_x and center_y
    center_yx = np.array(np.meshgrid(center_y, center_x)).T.reshape(-1, 2)

    # repeat for boxes = #priors
    center_yx = np.repeat(center_yx, num_priors, axis=0)

    # set xmin
    prior_boxes[:, 0] = center_yx[:, 1]
    # set ymin
    prior_boxes[:, 1] = center_yx[:, 0]
    # set xmax
    prior_boxes[:, 2] = center_yx[:, 1]
    # set ymax
    prior_boxes[:, 3] = center_yx[:, 0]

    # use box_widths and box_heights
    box_idx = 0
    for box_w, box_h in zip(box_widths, box_heights):
        prior_boxes[box_idx::num_priors, 0] -= box_w
        prior_boxes[box_idx::num_priors, 1] -= box_h
        prior_boxes[box_idx::num_priors, 2] += box_w
        prior_boxes[box_idx::num_priors, 3] += box_h
        box_idx += 1

    # clip the prior's coordinate such that it is within [0, 1]
    if clip:
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.), 1.)

    # set the variance.
    prior_boxes[:, 4:] = variance

    return prior_boxes


def _create_box_sizes(min_size, max_size, aspect_ratios):
    # prepare box widths and heights
    box_widths = []
    box_heights = []

    # the first prior: aspect_ratio = 1, size = min_size
    box_widths.append(min_size)
    box_heights.append(min_size)

    # the second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
    if max_size is not None:
        size = math.sqrt(min_size * max_size)
        box_widths.append(size)
        box_heights.append(size)

    # rest of priors
    for ar in aspect_ratios:
        if (math.fabs(ar - 1.) < 1e-6):
            continue

        box_widths.append(min_size * math.sqrt(ar))
        box_heights.append(min_size / math.sqrt(ar))

    return np.array(box_widths), np.array(box_heights)