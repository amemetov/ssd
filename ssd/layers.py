import json
import numpy as np

import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.initializers import Constant



class L2Normalize(Layer):
    """ l2-norm layer as described in the ParseNet

    # Arguments
        scale: from ParseNet doc:
            For example, we tried to normalize a feature s.t. l2-norm is 1,
            yet we can hardly train the network because the features become very small.
            However, if we normalize it to e.g. 10 or 20, the network begins to learn well.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first' ('th')
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last' ('tf').

    # Output shape
        Same shape as input.

    # References
        - [ParseNet: Looking Wider to See Better](http://cs.unc.edu/~wliu/papers/parsenet.pdf)
    """
    def __init__(self, scale, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)
        if K.image_dim_ordering() == 'tf':
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.scale = scale

    def build(self, input_shape):
        channel_dim = (input_shape[self.channel_axis],)

        self.gamma = self.add_weight(name='{}_gamma'.format(self.name),
                                     shape=channel_dim,
                                     initializer=Constant(value=self.scale),
                                     trainable=True)

        self.input_spec = [InputSpec(shape=input_shape)]
        self.built = True

    def call(self, inputs, **kwargs):
        output = K.l2_normalize(inputs, self.channel_axis)
        output *= self.gamma
        return output

    def get_config(self):
        config = {'scale': int(self.scale)}
        base_config = super(L2Normalize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        scale = config['scale']
        return cls(scale)


class PriorBox(Layer):
    """ Layer that generates the prior boxes of designated sizes and aspect ratios
        across all dimensions w*h.

    # Arguments
        img_size: size of the input image (w, h)
        min_size: minimum box size in pixels
        max_size: maximum box size in pixels
        aspect_ratios: boxes aspect ratios
        clip: whether to clip coordinates in the range [0, 1]

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first' ('th')
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last' ('tf').


    # Output shape
        3D tensor with shape:
        (samples, num_prior_boxes, 8)

    # References
        - [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
    """

    def __init__(self, prior_boxes, **kwargs):
        super(PriorBox, self).__init__(**kwargs)
        self.prior_boxes = prior_boxes

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        num_boxes = len(self.prior_boxes)
        return (batch_size, num_boxes, 8)

    def call(self, inputs, **kwargs):
        prior_boxes_tensor = K.variable(self.prior_boxes)
        prior_boxes_tensor = K.expand_dims(prior_boxes_tensor, 0)
        prior_boxes_tensor = K.tile(prior_boxes_tensor, [K.shape(inputs)[0], 1, 1])
        return prior_boxes_tensor

    def get_config(self):
        config = {'prior_boxes': json.dumps(self.prior_boxes.tolist())}
        base_config = super(PriorBox, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        prior_boxes = np.array(json.loads(config['prior_boxes']))
        return cls(prior_boxes)