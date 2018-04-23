import numpy as np

def loss(y_true, y_pred, num_classes, hard_neg_pos_ratio=3.0):
    loc_alpha = (hard_neg_pos_ratio + 1.) / 4.

    batch_size = np.shape(y_true)[0]
    num_boxes = np.shape(y_true)[1]

    # (batch_size, num_boxes)
    y_true_pb_gtb_matching = y_true[:, :, -8]
    num_pos = np.sum(y_true_pb_gtb_matching).astype(np.int)
    print('num_pos: {}'.format(num_pos))

    loc_loss = _loc_loss(y_true, y_pred, y_true_pb_gtb_matching)
    conf_loss = _conf_loss(y_true, y_pred, y_true_pb_gtb_matching, num_pos, batch_size, num_boxes, num_classes,
                           hard_neg_pos_ratio)

    total_loss = (conf_loss + loc_alpha * loc_loss) / num_pos

    print('loc_loss: {}'.format(loc_loss))

    return total_loss

def _loc_loss(y_true, y_pred, y_true_pb_gtb_matching):
    # extract loc classes and data
    y_true_loc = y_true[:, :, :4]
    y_pred_loc = y_pred[:, :, :4]

    # (batch_size, nb_boxes, )
    loc_loss = _smooth_l1_loss(y_true_loc, y_pred_loc)

    # (batch_size)
    loc_pos_loss = np.sum(y_true_pb_gtb_matching * loc_loss, axis=1)

    # scalar - sum for all images in batch
    loc_pos_loss = np.sum(loc_pos_loss)
    return loc_pos_loss


def _conf_loss(y_true, y_pred, y_true_pb_gtb_matching, num_pos, batch_size, num_boxes, num_classes, hard_neg_pos_ratio):
    conf_pos_loss = _calc_conf_pos_loss(y_true, y_pred, y_true_pb_gtb_matching, num_classes)
    conf_neg_loss = _calc_conf_neg_loss(y_true, y_pred, y_true_pb_gtb_matching, num_pos, batch_size, num_boxes,
                                        num_classes, hard_neg_pos_ratio)
    print('conf_pos_loss: {}'.format(conf_pos_loss))
    print('conf_neg_loss: {}'.format(conf_neg_loss))
    return conf_pos_loss + conf_neg_loss


def _calc_conf_pos_loss(y_true, y_pred, y_true_pb_gtb_matching, num_classes):
    # define conf indices excluding background
    conf_start_idx, conf_end_idx = _foreground_classes_indices(num_classes)
    y_true_classes = y_true[:, :, conf_start_idx:conf_end_idx]
    y_pred_classes = y_pred[:, :, conf_start_idx:conf_end_idx]

    # conf loss for all PriorBoxes
    # (batch_size, num_boxes)
    conf_loss = _softmax_loss(y_true_classes, y_pred_classes)

    # tensor of the shape (batch_size)
    # containing conf pos loss for each image (in the batch)
    # (batch_size)
    conf_pos_loss = np.sum(y_true_pb_gtb_matching * conf_loss, axis=1)

    # scalar
    return np.sum(conf_pos_loss)


def _calc_conf_neg_loss(y_true, y_pred, y_true_pb_gtb_matching, num_pos, batch_size, num_boxes, num_classes,
                        hard_neg_pos_ratio):
    # hard negative mining

    # tensor of the shape (batch_size)
    # containing # of not matching boxes for each image (in the batch)
    # clipped above using hard_neg_pos_ratio
    # num_neg = K.minimum(self.hard_neg_pos_ratio * num_pos, num_boxes - num_pos)

    # num_neg = np.cast(batch_size * num_boxes, dtype='float32') - num_pos
    num_neg = (batch_size * num_boxes - num_pos)
    num_neg = (np.minimum(hard_neg_pos_ratio * num_pos, num_neg)).astype(np.int)
    print('num_neg: {}'.format(num_neg))
    if num_neg == 0:
        return 0

    # get negatives
    negatives_indices = y_true_pb_gtb_matching == 0

    # tensor shape (total_num_neg, 4 + num_classes + 4 + 4)
    negatives_true = y_true[negatives_indices]
    negatives_pred = y_pred[negatives_indices]

    # define conf indices excluding background
    conf_start_idx, conf_end_idx = _foreground_classes_indices(num_classes)

    # (total_num_neg,)
    max_confs = np.max(negatives_pred[:, conf_start_idx:conf_end_idx], axis=-1)
    top_indices = np.argpartition(max_confs, -num_neg)[-num_neg:]  # top num_neg elements (not ordered)
    # max_confs = K.max(negatives_pred[:, conf_start_idx:conf_end_idx], axis=-1)
    # _, top_indices = tf.nn.top_k(max_confs, k=K.cast(num_neg, 'int32'))

    # get top negatives
    # (num_neg, 4+num_classes+8)
    negatives_true = negatives_true[top_indices]
    negatives_pred = negatives_pred[top_indices]

    # (num_neg,)
    # calc loss only for background class
    conf_neg_loss = _softmax_loss(negatives_true[:, 4], negatives_pred[:, 4])
    # scalar
    return np.sum(conf_neg_loss)

def _foreground_classes_indices(num_classes):
    # define conf indices excluding background
    start_idx = 4 + 1
    end_idx = start_idx + num_classes
    return start_idx, end_idx

def _smooth_l1_loss(y_true, y_pred):
    # https://arxiv.org/abs/1504.08083
    abs_loss = np.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred) ** 2
    otherwise_loss = abs_loss - 0.5
    l1_loss = np.where(abs_loss < 1.0, sq_loss, otherwise_loss)
    return np.sum(l1_loss, -1)


def _softmax_loss(y_true, y_pred):
    # prevent division by zero
    y_pred = np.maximum(y_pred, 1e-15)
    # return losses.categorical_crossentropy(y_true, y_pred)
    log_loss = -np.sum(y_true * np.log(y_pred), axis=-1)
    return log_loss