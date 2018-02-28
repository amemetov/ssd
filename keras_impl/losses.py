import keras.backend as K

class SsdLoss(object):
    def __init__(self, num_classes, num_variances, loc_alpha=1.0, hard_neg_pos_ratio=3.0,
                 background_class_id=0):
        self.num_classes = num_classes
        self.num_variances = num_variances
        self.loc_alpha = loc_alpha
        self.hard_neg_pos_ratio = hard_neg_pos_ratio
        self.background_class_id = background_class_id

    def loss(self, y_true, y_pred):
        """
        Compute the loss - see https://arxiv.org/abs/1512.02325

        # Arguments
            y_true: Ground Truth Boxes (gtb)
            tensor of shape (batch_size, num_boxes, 4 + num_classes + 4 + num_variances),
            where
            y_true[:, :, 0:4] - gtb loc (xmin, ymin, xmax, ymax)
            y_true[:, :, 4:4+num_classes] - one-hot-encoding classes
            y_true[:, :, -4-num_variances:-num_variances] - gtb prior box loc (xmin, ymin, xmax, ymax)
            y_true[:, :, -num_variances:] - gtb prior box variances

            y_pred: Predicted Boxes
            tensor of shape (batch_size, num_boxes, 4 + num_classes + 4 + num_variances),
            where
            y_true[:, :, 0:4] - predicted box loc (xmin, ymin, xmax, ymax)
            y_true[:, :, 4:4+num_classes] - one-hot-encoding predictions for classes
            y_true[:, :, -4-num_variances:-num_variances] - predicted prior box loc (xmin, ymin, xmax, ymax)
            y_true[:, :, -num_variances:] - predicted prior box variances

        # References
        - [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
        """


    def _smooth_l1_loss(self, y_true, y_pred):
        # https://arxiv.org/abs/1504.08083
        abs_loss = K.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        otherwise_loss = abs_loss - 0.5
        #TODO: implement without explicit TF
        l1_loss = K.tf.where(K.less(abs_loss, 1.0), sq_loss, otherwise_loss)
        return K.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred)

