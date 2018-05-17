import keras.backend as K
from keras import losses
import tensorflow as tf

class SsdLoss(object):
    def __init__(self, num_classes, hard_neg_pos_ratio=3):
        self.num_classes = num_classes
        self.loc_alpha = (hard_neg_pos_ratio + 1.) / 4.
        self.hard_neg_pos_ratio = hard_neg_pos_ratio

    def loss(self, y_true, y_pred):
        """
        Compute the loss - see https://arxiv.org/abs/1512.02325

        # Arguments
            y_true: Ground Truth Boxes (GTB)
            tensor of shape (batch_size, num_boxes, 4 + num_classes + 4 + 4),

            y_true[:, :, 0:4] - encoded GTB offsets in bbox locations (cx, cy, w, h) - see BBoxCodec.encode.
            y_true[:, :, 4:4+num_classes] - ground truth one-hot-encoding classes
            y_true[:, :, -8] - {0, 1} is the indicator for matching the current PriorBox to the GTB,
            not all row has GTB, often it is the background
            y_true[:, :, -7:] - 0 - is necessary only to have shape as y_pred's shape

            y_pred: Predicted Boxes
            tensor of shape (batch_size, num_boxes, 4 + num_classes + 4 + 4),
            where
            y_pred[:, :, 0:4] - predicted offsets in bbox locations (cx, cy, w, h) - see BBoxCodec.encode.
            y_pred[:, :, 4:4+num_classes] - one-hot-encoding predictions for classes
            y_pred[:, :, -8:-4] - PriorBox loc (xmin, ymin, xmax, ymax) - is used to decode predicted offsets
            y_pred[:, :, -4:] - PriorBox variances - may be used to decode predicted offsets

        # References
        - [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
        """

        batch_size = K.shape(y_true)[0]
        num_boxes = K.shape(y_true)[1]

        # PriorBox-GTB matching indicator - xij in the SSD paper
        # tensor of the shape (batch_size, num_boxes) containing {1, 0}
        y_matching_mask = y_true[:, :, -8]

        # tensor of the shape (batch_size)
        # containing # of matching boxes for each image (in the batch)
        num_pos = K.cast(K.sum(y_matching_mask, axis=1), dtype='int32')

        # tensors of the shape (batch_size)
        loc_loss = self._calc_loc_loss(y_true, y_pred, y_matching_mask, num_pos)
        conf_loss = self._calc_conf_loss(y_true, y_pred, y_matching_mask, num_pos, num_boxes)

        return K.sum(conf_loss + self.loc_alpha * loc_loss)# * tf.to_float(batch_size)

    def _calc_loc_loss(self, y_true, y_pred, y_matching_mask, num_pos):
        # extract loc classes and data
        y_true_loc = y_true[:, :, :4]
        y_pred_loc = y_pred[:, :, :4]

        # loc loss for all PriorBoxes
        loc_loss = self._smooth_l1_loss(y_true_loc, y_pred_loc)

        # 1D Tensor (batch_size, ) containing loc pos loss for each image (in the batch)
        loc_pos_loss = K.sum(y_matching_mask * loc_loss, axis=1)

        # average
        loc_pos_loss = loc_pos_loss / K.cast(num_pos, dtype='float32')

        # process zero-division
        loc_pos_loss = tf.where(tf.equal(num_pos, 0), tf.zeros_like(loc_pos_loss), loc_pos_loss)
        return loc_pos_loss

    def _calc_conf_loss(self, y_true, y_pred, y_matching_mask, num_pos, num_boxes):
        # define conf indices including background
        conf_start_idx, conf_end_idx = 4, 4 + self.num_classes

        # conf loss for all PriorBoxes
        # (batch_size, num_boxes)
        full_conf_loss = self._softmax_loss(y_true[:, :, conf_start_idx:conf_end_idx], y_pred[:, :, conf_start_idx:conf_end_idx])

        pos_indices_mask = y_matching_mask
        neg_indices_mask = self._mine_hard_examples(full_conf_loss, y_true, y_pred, y_matching_mask, num_pos, num_boxes)

        conf_pos_loss = K.sum(pos_indices_mask * full_conf_loss, axis=1)
        conf_neg_loss = K.sum(neg_indices_mask * full_conf_loss, axis=1)

        num_neg = K.cast(K.sum(neg_indices_mask, axis=1), dtype='int32')

        # average
        conf_pos_loss = conf_pos_loss / K.cast(num_pos, dtype='float32')
        conf_neg_loss = conf_neg_loss / K.cast(num_neg, dtype='float32')

        # process zero-division
        conf_pos_loss = tf.where(tf.equal(num_pos, 0), tf.zeros_like(conf_pos_loss), conf_pos_loss)
        conf_neg_loss = tf.where(tf.equal(num_neg, 0), tf.zeros_like(conf_neg_loss), conf_neg_loss)
        return conf_pos_loss + conf_neg_loss


    def _mine_hard_examples(self, full_conf_loss, y_true, y_pred, y_matching_mask, num_pos, num_boxes):
        # hard negative mining

        # tensor of the shape (batch_size)
        # containing # of not matching boxes for each image (in the batch)
        # clipped above using hard_neg_pos_ratio
        num_neg = K.minimum(self.hard_neg_pos_ratio * num_pos, num_boxes - num_pos)
        num_neg = K.maximum(num_neg, 1)
        #print('num_neg: {}'.format(num_neg))

        # (batch_size, num_boxes)
        all_neg_indices = K.cast(y_matching_mask == 0, dtype='float32')
        full_conf_neg_loss = full_conf_loss * all_neg_indices

        num_batch = K.shape(y_true)[0]
        top_indices_mask = tf.TensorArray(dtype=tf.float32, size=num_batch)

        # for each batch collect indices_mask
        shape = [num_boxes]
        cond = lambda b, _: tf.less(b, num_batch)
        def body(b, top_indices_mask):
            _, top_indices = tf.nn.top_k(full_conf_neg_loss[b], k=K.cast(num_neg[b], 'int32'))
            on_indices = top_indices[-num_neg[b]:]
            updates = tf.ones_like(on_indices, dtype='float32')
            batch_mask = tf.scatter_nd(on_indices, updates, shape)
            top_indices_mask = top_indices_mask.write(b, batch_mask)
            return [tf.add(b, 1), top_indices_mask]

        _, top_indices_mask = tf.while_loop(cond, body, [0, top_indices_mask])
        return top_indices_mask.stack()

    def _smooth_l1_loss(self, y_true, y_pred):
        # https://arxiv.org/abs/1504.08083
        abs_loss = K.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        otherwise_loss = abs_loss - 0.5
        l1_loss = tf.where(K.less(abs_loss, 1.0), sq_loss, otherwise_loss)
        return K.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        return losses.categorical_crossentropy(y_true, y_pred)

        # prevent division by zero
        # eps = 1e-15
        # y_pred = K.clip(y_pred, eps, 1. - eps)
        # log_loss = -K.sum(y_true * K.log(y_pred), axis=-1)
        # return log_loss

