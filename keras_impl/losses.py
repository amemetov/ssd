import keras.backend as K
from keras import losses
import tensorflow as tf

class SsdLoss(object):
    def __init__(self, num_classes, loc_alpha=1.0, hard_neg_pos_ratio=3.0,
                 background_class_id=0):
        self.num_classes = num_classes
        self.loc_alpha = loc_alpha
        self.hard_neg_pos_ratio = hard_neg_pos_ratio
        self.background_class_id = background_class_id

    def loss(self, y_true, y_pred):
        """
        Compute the loss - see https://arxiv.org/abs/1512.02325

        # Arguments
            y_true: Ground Truth Boxes (GTB)
            tensor of shape (batch_size, num_boxes, 4 + num_classes + 4 + 4),

            y_true[:, :, 0:4] - encoded GTB loc (xmin, ymin, xmax, ymax)
            y_true[:, :, 4:4+num_classes] - ground truth one-hot-encoding classes
            y_true[:, :, -8] - {0, 1} is the indicator for matching the current PriorBox to the GTB,
            not all row has GTB, often it is the background
            y_true[:, :, -7:] - 0 - is necessary only to have shape as y_pred's shape

            y_pred: Predicted Boxes
            tensor of shape (batch_size, num_boxes, 4 + num_classes + 4 + 4),
            where
            y_pred[:, :, 0:4] - predicted box loc (xmin, ymin, xmax, ymax)
            y_pred[:, :, 4:4+num_classes] - one-hot-encoding predictions for classes
            y_pred[:, :, -8:-4] - predicted prior box loc (xmin, ymin, xmax, ymax)
            y_pred[:, :, -4:] - predicted prior box variances

        # References
        - [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
        """

        batch_size = K.shape(y_true)[0]
        num_boxes = K.shape(y_true)[1]

        # PriorBox-GTB matching indicator - xij in the SSD paper
        # tensor of the shape (batch_size, num_boxes) containing {1, 0}
        y_true_pb_gtb_matching = y_true[:, :, -8]

        # tensor of the shape (batch_size)
        # containing # of matching boxes for each image (in the batch)
        # num_pos = K.sum(y_true_pb_gtb_matching, axis=-1)

        # scalar
        num_pos = K.sum(y_true_pb_gtb_matching)

        loc_loss = self._calc_loc_loss(y_true, y_pred, y_true_pb_gtb_matching)
        conf_loss = self._calc_conf_loss(y_true, y_pred, y_true_pb_gtb_matching, num_pos, batch_size, num_boxes)

        return (conf_loss + self.loc_alpha*loc_loss) / K.maximum(1.0, num_pos)


    def _calc_loc_loss(self, y_true, y_pred, y_true_pb_gtb_matching):
        # extract loc classes and data
        y_true_loc = y_true[:, :, :4]
        y_pred_loc = y_pred[:, :, :4]

        # loc loss for all PriorBoxes
        loc_loss = self._smooth_l1_loss(y_true_loc, y_pred_loc)

        # tensor of the shape (batch_size)
        # containing loc pos loss for each image (in the batch)
        loc_pos_loss = K.sum(y_true_pb_gtb_matching * loc_loss, axis=1)
        loc_pos_loss = K.sum(loc_pos_loss)
        return loc_pos_loss

    def _calc_conf_loss(self, y_true, y_pred, y_true_pb_gtb_matching, num_pos, batch_size, num_boxes):
        conf_pos_loss = self._calc_conf_pos_loss(y_true, y_pred, y_true_pb_gtb_matching)
        conf_neg_loss = self._calc_conf_neg_loss(y_true, y_pred, y_true_pb_gtb_matching, num_pos, batch_size, num_boxes)
        return K.sum(conf_pos_loss) + K.sum(conf_neg_loss)

    def _calc_conf_pos_loss(self, y_true, y_pred, y_true_pb_gtb_matching):
        y_true_classes = y_true[:, :, 4:4 + self.num_classes]
        y_pred_classes = y_pred[:, :, 4:4 + self.num_classes]

        # conf loss for all PriorBoxes
        conf_loss = self._softmax_loss(y_true_classes, y_pred_classes)

        # tensor of the shape (batch_size)
        # containing conf pos loss for each image (in the batch)
        return K.sum(y_true_pb_gtb_matching * conf_loss, axis=1)

    def _calc_conf_neg_loss(self, y_true, y_pred, y_true_pb_gtb_matching, num_pos, batch_size, num_boxes):
        # hard negative mining

        # tensor of the shape (batch_size)
        # containing # of not matching boxes for each image (in the batch)
        # clipped above using hard_neg_pos_ratio
        # num_neg = K.minimum(self.hard_neg_pos_ratio * num_pos, num_boxes - num_pos)

        num_neg = K.cast(batch_size * num_boxes, dtype='float32') - num_pos
        num_neg = K.minimum(self.hard_neg_pos_ratio * num_pos, num_neg)
        if num_neg == 0:
            return 0

        # get negatives
        negatives_indices = K.equal(y_true[:, :, -8], 0) # y_true[:, :, -8] == 0

        # tensor shape (total_num_neg, 4 + num_classes + 4 + 4)
        negatives_true = tf.boolean_mask(y_true, negatives_indices) # y_true[negatives_indices]
        negatives_pred = tf.boolean_mask(y_pred, negatives_indices) # y_pred[negatives_indices]

        # define conf indices excluding background
        conf_start_idx = 4 + self.background_class_id + 1
        conf_end_idx = conf_start_idx + self.num_classes - 1

        # (total_num_neg,)
        #max_confs = np.max(negatives_pred[:, conf_start_idx:conf_end_idx], axis=-1)
        #top_indices = np.argpartition(max_confs, -num_neg)[-num_neg:] # top num_neg elements (not ordered)
        max_confs = K.max(negatives_pred[:, conf_start_idx:conf_end_idx], axis=-1)
        _, top_indices = tf.nn.top_k(max_confs, k=K.cast(num_neg, 'int32'))

        # get top negatives
        negatives_true = K.gather(negatives_true, top_indices) # negatives_true[top_indices]
        negatives_pred = K.gather(negatives_pred, top_indices) # negatives_pred[top_indices]

        return self._softmax_loss(negatives_true, negatives_pred)

    def _smooth_l1_loss(self, y_true, y_pred):
        # https://arxiv.org/abs/1504.08083
        abs_loss = K.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        otherwise_loss = abs_loss - 0.5
        #TODO: implement without explicit TF
        l1_loss = K.tf.where(K.less(abs_loss, 1.0), sq_loss, otherwise_loss)
        return K.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        # prevent division by zero
        y_pred = K.maximum(y_pred, 1e-15)
        return losses.categorical_crossentropy(y_true, y_pred)

