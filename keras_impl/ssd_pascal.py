import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Cropping2D, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam

from keras.applications import VGG16

def SSD300(base_net_fine_tune_start_layer='conv4_1'):
    # VGG16 - Input size must be at least 48x48;
    # SSD 300 - input size 300x300
    input_dim = (300, 300, 3)

    image_input = Input(shape=input_dim, name='image_input')

    base_net = VGG16(include_top=False, weights='imagenet',
                      input_tensor=image_input, input_shape=input_dim)

    _freezeBaseNet(base_net, base_net_fine_tune_start_layer)

    predictions = addExtraLayers(base_net, use_bn=True)

    model = Model(inputs=[image_input], outputs=[predictions])

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
def addExtraLayers(base_net, use_bn=True):
    # Add additional convolutional layers.
    # 19 x 19
    #from_layer = base_net#net.keys()[-1]
    x = base_net.outputs[0]

    print(x)

    # 10 x 10
    x = ConvBNLayer(x, "conv6_1", use_bn, 256, 1, 1, 'same')


    x = ConvBNLayer(x, "conv6_2", use_bn, 512, 3, 2, 'valid')

    # 5 x 5
    x = ConvBNLayer(x, "conv7_1", use_bn, 128, 1, 1, 'same')

    x = ConvBNLayer(x, "conv7_2", use_bn, 256, 3, 2, 'valid')

    # 3 x 3
    x = ConvBNLayer(x, "conv8_1", use_bn, 128, 1, 1, 'same')

    x = ConvBNLayer(x, "conv8_2", use_bn, 256, 3, 1, 'same')

    # 1 x 1
    x = ConvBNLayer(x, "conv9_1", use_bn, 128, 1, 1, 'same')

    x = ConvBNLayer(x, "conv9_2", use_bn, 256, 3, 1, 'same')

    return x


def ConvBNLayer(net, layer_name, use_bn,
                num_output, kernel_size, stride, pad,
                dilation=1):
    x = Convolution2D(filters=num_output, kernel_size=kernel_size,
                      strides=stride,
                      padding=pad,
                      dilation_rate=dilation,
                      activation='relu',
                      name=layer_name)(net)
    if use_bn:
        x = BatchNormalization()(x)

    return x


def _freezeBaseNet(conv_base, fine_tune_start_layer='conv4_1'):
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == fine_tune_start_layer:
            set_trainable = True

        layer.trainable = set_trainable