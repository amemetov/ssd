import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.initializers import Constant

from .prior_box import doCreatePriorBoxes



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

    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 variance=[0.1], clip=True, offset=0.5,
                 **kwargs):
        super(PriorBox, self).__init__(**kwargs)

        if min_size <= 0:
            raise ValueError('min_size should be positive')

        if max_size is not None and max_size < min_size:
            raise ValueError('max_size must be greater than min_size')

        self.img_w, self.img_h = img_size
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios
        self.variance = variance
        self.clip = clip
        self.offset = offset

        if K.image_dim_ordering() == 'tf':
            self.w_axis, self.h_axis = 2, 1
        else:
            self.w_axis, self.h_axis = 3, 2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        num_priors = len(self.aspect_ratios)
        layer_w = input_shape[self.w_axis]
        layer_h = input_shape[self.h_axis]
        num_boxes = num_priors * layer_w * layer_h
        num_variances = len(self.variance)
        return (batch_size, num_boxes, 4 + num_variances)

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)

        layer_w, layer_h = input_shape[self.w_axis], input_shape[self.h_axis]

        prior_boxes = doCreatePriorBoxes(self.img_w, self.img_h, layer_w, layer_h,
                                         self.min_size, self.max_size, self.aspect_ratios,
                                         self.variance, self.clip, self.offset)

        prior_boxes_tensor = K.variable(prior_boxes)
        prior_boxes_tensor = K.expand_dims(prior_boxes_tensor, 0)
        prior_boxes_tensor = K.tile(prior_boxes_tensor, [K.shape(inputs)[0], 1, 1])
        return prior_boxes_tensor