import math

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

def gen_prior_boxes_config(img_w, img_h, layers_size, aspect_ratios,
                           flip=True, min_ratio=10, max_ratio = 90):
                           #min_ratio=20, max_ratio = 90):
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
    if flip:
        aspect_ratios = [1.0] + [f(x) for x in aspect_ratios for f in (lambda x: x, lambda x: 1. / x)]
    else:
        aspect_ratios = [1.0] + aspect_ratios

    return {'layer_width': layer_w, 'layer_height': layer_h,
            'num_prior': len(aspect_ratios),
            'min_size': min_size, 'max_size': max_size,
            'aspect_ratios': aspect_ratios}


def create_prior_boxes(box_config):
    for layer_config in box_config:
        layer_width = layer_config['layer_width']
        layer_height = layer_config['layer_height']
        min_sizes = layer_config['min_size']
        max_sizes = layer_config['max_size']
        aspect_ratios_orig = layer_config['aspect_ratios']
        flip = layer_config['flip']
        clip = layer_config['clip']
        variance = layer_config['variance']

        # build aspect_ratios
        aspect_ratios = []
        # first is always 1
        aspect_ratios.append(1.0)
        for ar in aspect_ratios_orig:
            aspect_ratios.append(ar)
            if flip:
                aspect_ratios.append(1.0 / ar)

        num_priors = len(aspect_ratios) * len(min_sizes)

        if len(max_sizes) > 0:
            assert len(max_sizes) == len(min_sizes), "len(max_sizes) == len(min_sizes)"
            num_priors += len(max_sizes)

        if len(variance) > 1:
            # Must and only provide 4 variance.
            assert len(variance) == 4, "Provide only 4 variance"
        elif len(variance) == 1:
            # Set default to 0.1.
            variance = [0.1]

        # img_w, img_h
        # step_w, step_h
        # offset






