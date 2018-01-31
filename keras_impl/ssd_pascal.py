import numpy as np

from keras.models import Model
from keras.layers import Input, Cropping2D, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization
from keras.layers import merge
from keras.optimizers import Adam

# from keras.applications import VGG16
from .vgg import VGG16

from .layers import L2Normalize

# layers_config = [
#     {'layer': 'conv4_3', 'normalization': 20, 'layer_width': 38, 'layer_height': 38, 'num_prior': 3,
#      'min_size':  30.0, 'max_size': None, 'aspect_ratios': [1.0, 2.0, 1/2.0]},
#
#     {'layer': 'fc7', 'normalization': -1, 'layer_width': 19, 'layer_height': 19, 'num_prior': 6,
#      'min_size':  60.0, 'max_size': 114.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
#
#     {'layer': 'conv6_2', 'normalization': -1, 'layer_width': 10, 'layer_height': 10, 'num_prior': 6,
#      'min_size': 114.0, 'max_size': 168.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
#
#     {'layer': 'conv7_2', 'normalization': -1, 'layer_width':  5, 'layer_height':  5, 'num_prior': 6,
#      'min_size': 168.0, 'max_size': 222.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
#
#     {'layer': 'conv8_2', 'normalization': -1, 'layer_width':  3, 'layer_height':  3, 'num_prior': 6,
#      'min_size': 222.0, 'max_size': 276.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
#
#     {'layer': 'conv9_2', 'normalization': -1, 'layer_width':  1, 'layer_height':  1, 'num_prior': 6,
#      'min_size': 276.0, 'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
# ]

layers_config = [
    {'layer': 'conv4_3', 'normalization': 20, 'layer_width': 38, 'layer_height': 38, 'num_prior': 4,
     'min_size': 30.0, 'max_size': 60.0, 'aspect_ratios': [1.0, 1.0, 2.0, 0.5]},

    {'layer': 'fc7', 'normalization': -1, 'layer_width': 19, 'layer_height': 19, 'num_prior': 6,
     'min_size': 60.0, 'max_size': 111.0, 'aspect_ratios': [1.0, 1.0, 2.0, 0.5, 3, 1/3.0]},

    {'layer': 'conv6_2', 'normalization': -1, 'layer_width': 10, 'layer_height': 10, 'num_prior': 6,
     'min_size': 111.0, 'max_size': 162.0, 'aspect_ratios': [1.0, 1.0, 2.0, 0.5, 3, 1/3.0]},

    {'layer': 'conv7_2', 'normalization': -1, 'layer_width': 5, 'layer_height': 5, 'num_prior': 6,
     'min_size': 162.0, 'max_size': 213.0, 'aspect_ratios': [1.0, 1.0, 2.0, 0.5, 3, 1/3.0]},

    {'layer': 'conv8_2', 'normalization': -1, 'layer_width': 3, 'layer_height': 3, 'num_prior': 4,
     'min_size': 213.0, 'max_size': 264.0, 'aspect_ratios': [1.0, 1.0, 2.0, 0.5]},

    {'layer': 'conv9_2', 'normalization': -1, 'layer_width': 1, 'layer_height': 1, 'num_prior': 4,
     'min_size': 264.0, 'max_size': 315.0, 'aspect_ratios': [1.0, 1.0, 2.0, 0.5]}
]

prior_variance = [0.1, 0.1, 0.2, 0.2]

def SSD300(base_net_name='vgg16', freeze_layers=None, use_bn=False):
    if base_net_name not in {'vgg16', None}:
        raise ValueError('The `base_net_name` argument should be either '
                         '`vgg16` or (other nets will be added in the future).')

    input_dim = (300, 300, 3)

    ssd_net = {}

    ssd_net['image_input'] = Input(shape=input_dim, name='image_input')

    base_net_model = VGG16(ssd_net, ssd_net['image_input'], input_shape=input_dim,
                           include_top=False, weights='imagenet')

    addExtraLayers(ssd_net, base_net_model, use_bn=use_bn)

    num_classes = 21

    mbox_layers = CreateMultiBoxHead(ssd_net, input_dim, num_classes=num_classes,
                                     layers_config=layers_config,
                                     flip=True, clip=False, prior_variance=prior_variance, offset=0.5,
                                     kernel_size=3, pad='same', #pad=1,
                                     use_bn=use_bn
                                     )

    ssd_net['predictions'] = merge(mbox_layers, mode='concat', concat_axis=2, name='predictions')

    model = Model(inputs=[ssd_net['image_input']], outputs=[ssd_net['predictions']])

    _freezeLayers(model, freeze_layers)

    return model


    # AddExtraLayers(net, use_bn, lr_mult=lr_mult)
    #
    # mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
    #                                  use_bn=use_bn, min_sizes=min_sizes, max_sizes=max_sizes,
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
    x = ConvBNLayer(ssd_net, x, "conv6_1", 256, 1, 'valid', 1, use_bn)
    x = ConvBNLayer(ssd_net, x, "conv6_2", 512, 3, 'same', 2, use_bn)

    # 5 x 5
    x = ConvBNLayer(ssd_net, x, "conv7_1", 128, 1, 'valid', 1, use_bn)
    x = ConvBNLayer(ssd_net, x, "conv7_2", 256, 3, 'same', 2, use_bn)

    # 3 x 3
    x = ConvBNLayer(ssd_net, x, "conv8_1", 128, 1, 'valid', 1, use_bn)
    x = ConvBNLayer(ssd_net, x, "conv8_2", 256, 3, 'valid', 1, use_bn)

    # 1 x 1
    x = ConvBNLayer(ssd_net, x, "conv9_1", 128, 1, 'valid', 1, use_bn)
    x = ConvBNLayer(ssd_net, x, "conv9_2", 256, 3, 'valid', 1, use_bn)


def ConvBNLayer(net, in_tensor, result_layer_name,
                num_output, kernel_size, pad, stride, dilation=1, use_bn=False):
    net[result_layer_name] = x = Conv2D(filters=num_output, kernel_size=kernel_size,
                                        strides=stride,
                                        padding=pad,
                                        dilation_rate=dilation,
                                        activation='relu',
                                        name=result_layer_name)(in_tensor)
    if use_bn:
        net[result_layer_name + '_bn'] = x = BatchNormalization(name=result_layer_name + '_bn')(x)

    return x


def CreateMultiBoxHead(net, img_size, num_classes,
                       layers_config,
                       flip=True, clip=False, prior_variance=[0.1, 0.1, 0.2, 0.2], offset=0.5,
                       kernel_size=1, pad='valid', #pad=0,
                       use_bn=True):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"


    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    # for i in range(0, num):
    for layer_config in layers_config:
        layer = layer_config['layer']
        normalization = layer_config['normalization']
        num_priors_per_location = layer_config['num_prior']
        min_size = layer_config['min_size']
        max_size = layer_config['max_size']
        aspect_ratios = layer_config['aspect_ratios']

        # Get the normalize value.
        if normalization != -1:
            norm_name = "{}_norm".format(layer)
            net[norm_name] = L2Normalize(scale=normalization, name=norm_name)(net[layer])
            layer = norm_name


        # Create location prediction layer.
        loc_name = "{}_mbox_loc".format(layer)
        num_loc_output = num_priors_per_location * 4
        x = ConvBNLayer(net, net[layer], loc_name,
                    num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, use_bn=use_bn)
        #permute_name = "{}_perm".format(loc_name)
        #net[permute_name] = L.Permute(net[loc_name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(loc_name)
        net[flatten_name] = Flatten(name=flatten_name)(x)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        conf_name = "{}_mbox_conf".format(layer)
        num_conf_output = num_priors_per_location * num_classes
        x = ConvBNLayer(net, net[layer], conf_name,
                    num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, use_bn=use_bn)
        #permute_name = "{}_perm".format(conf_name)
        #net[permute_name] = L.Permute(net[conf_name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(conf_name)
        net[flatten_name] = Flatten(name=flatten_name)(x)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        priorbox_name = "{}_mbox_priorbox".format(layer)
        net[priorbox_name] = PriorBox(net, net[layer], priorbox_name,
                                      img_size=img_size,
                                      min_size=min_size, max_size=max_size, aspect_ratios=aspect_ratios,
                                      flip=flip, clip=clip, variances=prior_variance, offset=offset)
        # if step:
        #     net.update(name, {'step': step})
        priorbox_layers.append(net[priorbox_name])


    # Concatenate priorbox, loc, and conf layers.
    net['mbox_loc'] = merge(loc_layers, mode='concat', concat_axis=1, name='mbox_loc')
    net['mbox_conf'] = merge(conf_layers, mode='concat', concat_axis=1, name='mbox_conf')
    net['mbox_priorbox'] = merge(priorbox_layers, mode='concat', concat_axis=1, name='mbox_priorbox')

    return [net['mbox_loc'], net['mbox_conf'], net['mbox_priorbox']]


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
