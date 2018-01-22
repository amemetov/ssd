import math
import numpy as np

# https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_pascal.py
# https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp

default_img_width, default_img_height = 300, 300

# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# conv9_2 ==> 1 x 1
default_layers_size=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
default_aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
default_steps = [8, 16, 32, 64, 100, 300]

# L2 normalize conv4_3.
default_normalizations = [20, -1, -1, -1, -1, -1]

# variance used to encode/decode prior bboxes.
default_prior_variance = [0.1, 0.1, 0.2, 0.2]

default_flip = True
default_clip = True

def gen_prior_boxes_config(img_w, img_h, layers_size, aspect_ratios,
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
    layer_w, layer_h = layers_size[0]
    box_config.append(_create_config_item(layer_w, layer_h, min_size, max_size, aspect_ratios[0], flip))
    ratio = min_ratio
    for layer_size, ar in zip(layers_size[1:], aspect_ratios[1:]):
        layer_w, layer_h = layer_size
        min_size = img_min_dim * ratio / 100.
        max_size = img_min_dim * (ratio + ratio_step) / 100.
        box_config.append(_create_config_item(layer_w, layer_h, min_size, max_size, ar, flip))
        ratio += ratio_step

    return box_config

def _create_config_item(layer_w, layer_h, min_size, max_size, aspect_ratios, flip):
    first_and_second_ars = [1., 1.]
    if flip:
        aspect_ratios = first_and_second_ars + [f(x) for x in aspect_ratios for f in (lambda x: x, lambda x: 1. / x)]
    else:
        aspect_ratios = first_and_second_ars + aspect_ratios

    return {'layer_width': layer_w, 'layer_height': layer_h,
            'num_prior': len(aspect_ratios),
            'min_size': min_size, 'max_size': max_size,
            'aspect_ratios': aspect_ratios}


def create_prior_boxes(img_w, img_h, box_config, variance, clip=True):
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

        top_data = np.zeros(num_boxes * 8)

        idx = 0
        for h in range(0, layer_h):
            for w in range(0, layer_w):
                center_x = w * step_w
                center_y = h * step_h

                # first prior: aspect_ratio = 1, size = min_size
                box_w = box_h = min_size

                # xmin
                top_data[idx] = (center_x - box_w / 2.) / img_w
                # ymin
                top_data[idx + 1] = (center_y - box_h / 2.) / img_h
                # xmax
                top_data[idx + 2] = (center_x + box_w / 2.) / img_w
                # ymax
                top_data[idx + 3] = (center_y + box_h / 2.) / img_h
                idx += 4

                # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                box_w = box_h = math.sqrt(min_size * max_size)
                # xmin
                top_data[idx] = (center_x - box_w / 2.) / img_w
                # ymin
                top_data[idx + 1] = (center_y - box_h / 2.) / img_h
                # xmax
                top_data[idx + 2] = (center_x + box_w / 2.) / img_w
                # ymax
                top_data[idx + 3] = (center_y + box_h / 2.) / img_h
                idx += 4

                # rest of priors
                for ar in aspect_ratios:
                    if (math.fabs(ar - 1.) < 1e-6):
                        continue

                    box_w = min_size * math.sqrt(ar)
                    box_h = min_size / math.sqrt(ar)
                    # xmin
                    top_data[idx] = (center_x - box_w / 2.) / img_w
                    # ymin
                    top_data[idx + 1] = (center_y - box_h / 2.) / img_h
                    # xmax
                    top_data[idx + 2] = (center_x + box_w / 2.) / img_w
                    # ymax
                    top_data[idx + 3] = (center_y + box_h / 2.) / img_h
                    idx += 4


        print('idx = {0}'.format(idx))

        dim = num_boxes * 4
        print('dim = {0}'.format(dim))

        # clip the prior's coordidate such that it is within [0, 1]
        if clip:
            for d in range(0, dim):
                top_data[d] = min(max(top_data[d], 0.), 1.)

        # set the variance.
        count = dim
        for h in range(0, layer_h):
            for w in range(0, layer_w):
                for i in range(0, num_priors):
                    for j in range(0, 4):
                        top_data[count] = variance[j]
                        count += 1

        print('count = {0}'.format(count))

        return top_data
