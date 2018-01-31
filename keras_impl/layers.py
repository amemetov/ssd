import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.initializers import Constant



class L2Normalize(Layer):
    """ l2-norm layer as described in the ParseNet
    http://cs.unc.edu/~wliu/papers/parsenet.pdf

    # Arguments
        scale: from ParseNet doc:
            For example, we tried to normalize a feature s.t. l2-norm is 1,
            yet we can hardly train the network because the features become very small.
            However, if we normalize it to e.g. 10 or 20, the network begins to learn well.

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

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.channel_axis)
        output *= self.gamma
        return output
