import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Cropping2D, Lambda
from keras.layers import Conv2D, AtrousConv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam

# from keras.applications import VGG16
from .vgg import VGG16


def SSD300(base_net_name='vgg16', freeze_layers=None, use_bn=False):
    if base_net_name not in {'vgg16', None}:
        raise ValueError('The `base_net_name` argument should be either '
                         '`vgg16` or (other nets will be added in the future).')

    input_dim = (300, 300, 3)

    ssd_net = {}

    ssd_net['image_input'] = Input(shape=input_dim, name='image_input')

    base_net_model = VGG16(ssd_net, ssd_net['image_input'], input_shape=input_dim,
                           include_top=False, weights='imagenet')

    # predictions = addExtraLayers(base_net, use_bn=use_bn)
    addExtraLayers(ssd_net, base_net_model, use_bn=use_bn)

    ssd_net['predictions'] = ssd_net['conv9_2']

    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']

    # mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
    #                                  use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
    #                                  aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
    #                                  num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
    #                                  prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

    model = Model(inputs=[ssd_net['image_input']], outputs=[ssd_net['predictions']])

    _freezeLayers(model, freeze_layers)

    return model


    # AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)
    #
    # mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
    #                                  use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
    #                                  aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
    #                                  num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
    #                                  prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)
    #
    # # Create the MultiBoxLossLayer.
    # name = "mbox_loss"
    # mbox_layers.append(net.label)
    # net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
    #                            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
    #                            propagate_down=[True, True, False, False])


# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def addExtraLayers(ssd_net, base_net_model, use_bn=True, use_dropout=False):
    # Add additional convolutional layers.

    x = base_net_model.layers[-1].output

    # 19 x 19
    # FC6
    ssd_net['fc6'] = x = Conv2D(1024, 3, padding='same', dilation_rate=(6, 6), activation='relu', name='fc6')(x)
    if use_dropout:
        ssd_net['drop6'] = x = Dropout(0.5, name='drop6')(x)

    # FC7
    ssd_net['fc7'] = x = Conv2D(1024, 1, padding='same', activation='relu', name='fc7')(x)
    if use_dropout:
        ssd_net['drop7'] = x = Dropout(0.5, name='drop7')(x)

    # 10 x 10
    x = ConvBNLayer(ssd_net, x, "conv6_1", use_bn, 256, 1, 'valid', 1)
    x = ConvBNLayer(ssd_net, x, "conv6_2", use_bn, 512, 3, 'same', 2)

    # 5 x 5
    x = ConvBNLayer(ssd_net, x, "conv7_1", use_bn, 128, 1, 'valid', 1)
    x = ConvBNLayer(ssd_net, x, "conv7_2", use_bn, 256, 3, 'same', 2)

    # 3 x 3
    x = ConvBNLayer(ssd_net, x, "conv8_1", use_bn, 128, 1, 'valid', 1)
    x = ConvBNLayer(ssd_net, x, "conv8_2", use_bn, 256, 3, 'valid', 1)

    # 1 x 1
    x = ConvBNLayer(ssd_net, x, "conv9_1", use_bn, 128, 1, 'valid', 1)
    x = ConvBNLayer(ssd_net, x, "conv9_2", use_bn, 256, 3, 'valid', 1)


def ConvBNLayer(net, tensor, layer_name, use_bn,
                num_output, kernel_size, pad, stride,
                dilation=1):
    net[layer_name] = x = Conv2D(filters=num_output, kernel_size=kernel_size,
                                 strides=stride,
                                 padding=pad,
                                 dilation_rate=dilation,
                                 activation='relu',
                                 name=layer_name)(tensor)
    if use_bn:
        net[layer_name + '_bn'] = x = BatchNormalization(name=layer_name + '_bn')(x)

    return x


def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
                       use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
                       use_scale=True, min_sizes=[], max_sizes=[], prior_variance=[0.1],
                       aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
                       flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
                       conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer],
                                             scale_filler=dict(type="constant", value=normalizations[i]),
                                             across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                            num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                    num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                    num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                               clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                        num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers


def _freezeLayers(model, freeze_layers):
    model.trainable = True
    if freeze_layers is None:
        freeze_layers = ['conv1_1', 'conv1_2', 'pool1',
                         'conv2_1', 'conv2_2', 'pool2',
                         'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
                         # 'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
                         # 'conv5_1', 'conv5_2', 'conv5_3', 'pool5'
                         ]
    for layer in model.layers:
        layer.trainable = layer.name not in freeze_layers

# def _freezeBaseNet(conv_base, fine_tune_start_layer='conv4_1'):
#     conv_base.trainable = True
#     set_trainable = False
#     for layer in conv_base.layers:
#         if layer.name == fine_tune_start_layer:
#             set_trainable = True
#
#         layer.trainable = set_trainable
