import numpy as np

def loss(y_true, y_pred, num_classes, hard_neg_pos_ratio=3.0):
    loc_alpha = (hard_neg_pos_ratio + 1.) / 4.

    batch_size = np.shape(y_true)[0]
    num_boxes = np.shape(y_true)[1]

    # (batch_size, num_boxes)
    y_true_pb_gtb_matching = y_true[:, :, -8]
    #num_pos = np.sum(y_true_pb_gtb_matching).astype(np.int)
    num_pos = np.sum(y_true_pb_gtb_matching, axis=1).astype(np.int)
    print('num_pos: {}'.format(num_pos))

    loc_loss = _loc_loss(y_true, y_pred, y_true_pb_gtb_matching)
    conf_loss = _conf_loss(y_true, y_pred, y_true_pb_gtb_matching, num_pos, num_boxes, num_classes, hard_neg_pos_ratio)

    total_num_pos = np.sum(num_pos)

    if total_num_pos == 0:
        return 0

    total_loss = (conf_loss + loc_alpha * loc_loss) / total_num_pos
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
    return np.sum(loc_pos_loss)

def _conf_loss(y_true, y_pred, y_true_pb_gtb_matching, num_pos, num_boxes, num_classes, hard_neg_pos_ratio):
    # conf_pos_loss = _calc_conf_pos_loss(full_conf_loss, y_true_pb_gtb_matching)
    # conf_neg_loss = _calc_conf_neg_loss(full_conf_loss, y_true, y_pred, y_true_pb_gtb_matching,
    #                                    num_pos, num_boxes, hard_neg_pos_ratio)

    conf_start_idx, conf_end_idx = _classes_indices(num_classes)
    # conf loss for all PriorBoxes
    # (batch_size, num_boxes)
    full_conf_loss = _softmax_loss(y_true[:, :, conf_start_idx:conf_end_idx], y_pred[:, :, conf_start_idx:conf_end_idx])

    pos_indices = y_true_pb_gtb_matching
    neg_indices = _mine_hard_examples(full_conf_loss, y_true, y_pred, y_true_pb_gtb_matching,
                                    num_pos, num_boxes, hard_neg_pos_ratio)
    conf_pos_loss = np.sum(pos_indices * full_conf_loss, axis=1)
    conf_neg_loss = np.sum(neg_indices * full_conf_loss, axis=1)

    print('conf_pos_loss: {}'.format(conf_pos_loss))
    print('conf_neg_loss: {}'.format(conf_neg_loss))

    return np.sum(conf_pos_loss + conf_neg_loss)

def _mine_hard_examples(full_conf_loss, y_true, y_pred, y_true_pb_gtb_matching,
                        num_pos, num_boxes, hard_neg_pos_ratio):
    # hard negative mining

    # tensor of the shape (batch_size)
    # containing # of not matching boxes for each image (in the batch)
    # clipped above using hard_neg_pos_ratio
    num_neg = (num_boxes - num_pos)
    num_neg = (np.minimum(hard_neg_pos_ratio * num_pos, num_neg)).astype(np.int)
    print('num_neg: {}'.format(num_neg))

    #(batch_size, num_boxes)
    all_neg_indices = y_true_pb_gtb_matching == 0
    full_conf_neg_loss = full_conf_loss * all_neg_indices
    #print('full_conf_neg_loss: {}'.format(full_conf_neg_loss))

    top_indices = np.argsort(full_conf_neg_loss, axis=1)
    top_indices_mask = np.zeros_like(top_indices)

    # for each batch
    num_batch = np.shape(y_true)[0]
    for b in range(num_batch):
        top_indices_mask[b][top_indices[b, -num_neg[b]:]] = 1

    return top_indices_mask

def _calc_conf_pos_loss(full_conf_loss, y_true_pb_gtb_matching):
    # tensor of the shape (batch_size)
    # containing conf pos loss for each image (in the batch)
    # (batch_size)
    conf_pos_loss = np.sum(y_true_pb_gtb_matching * full_conf_loss, axis=1)

    # scalar
    return np.sum(conf_pos_loss)


def _calc_conf_neg_loss(full_conf_loss, y_true, y_pred, y_true_pb_gtb_matching,
                        num_pos, num_boxes, hard_neg_pos_ratio):
    # hard negative mining

    # tensor of the shape (batch_size)
    # containing # of not matching boxes for each image (in the batch)
    # clipped above using hard_neg_pos_ratio
    num_neg = (num_boxes - num_pos)
    num_neg = (np.minimum(hard_neg_pos_ratio * num_pos, num_neg)).astype(np.int)
    print('num_neg: {}'.format(num_neg))

    # get negatives
    #negatives_indices = y_true_pb_gtb_matching == 0
    #print('negatives_indices: {}'.format(negatives_indices))
    #print('negatives_indices.shape: {}'.format(negatives_indices.shape))

    # tensor shape (total_num_neg, 4 + num_classes + 4 + 4)
    #negatives_true = y_true[negatives_indices]
    #negatives_pred = y_pred[negatives_indices]

    # define conf indices excluding background
    #print('negatives_true: {}'.format(negatives_true.shape))
    #print('negatives_pred: {}'.format(negatives_pred.shape))
    #neg_conf_loss = _softmax_loss(negatives_true[:, conf_start_idx:conf_end_idx],
    #                                   negatives_pred[:, conf_start_idx:conf_end_idx])

    #(batch_size, num_boxes)
    y_true_pb_gtb_matching_not = y_true[:, :, -8] == 0
    conf_neg_loss = y_true_pb_gtb_matching_not * full_conf_loss

    # (total_num_neg,)
    #max_confs = np.max(negatives_pred[:, conf_start_idx:conf_end_idx], axis=-1)
    max_confs = conf_neg_loss
    top_indices = np.argpartition(max_confs, -num_neg)#[:, -num_neg:]  # top num_neg elements (not ordered)
    top_indices = top_indices[:, -num_neg:]  # top num_neg elements (not ordered)
    # max_confs = K.max(negatives_pred[:, conf_start_idx:conf_end_idx], axis=-1)
    # _, top_indices = tf.nn.top_k(max_confs, k=K.cast(num_neg, 'int32'))

    # get top negatives
    # (num_neg, 4+num_classes+8)
    #negatives_true = negatives_true[top_indices]
    #negatives_pred = negatives_pred[top_indices]
    negatives_true = y_true[top_indices]
    negatives_pred = y_pred[top_indices]

    # (num_neg,)
    # calc loss only for background class
    #conf_neg_loss = _softmax_loss(negatives_true[:, 4], negatives_pred[:, 4])

    conf_neg_loss = full_conf_loss


    # scalar
    return np.sum(conf_neg_loss)

def _classes_indices(num_classes):
    # define conf indices including background
    start_idx = 4
    end_idx = 4 + num_classes
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
    log_loss = -np.sum(y_true * np.log(y_pred), axis=-1)
    return log_loss