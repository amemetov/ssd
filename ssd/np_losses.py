import numpy as np

def loss(y_true, y_pred, num_classes, hard_neg_pos_ratio=3.0):
    loc_alpha = (hard_neg_pos_ratio + 1.) / 4.

    batch_size = np.shape(y_true)[0]
    num_boxes = np.shape(y_true)[1]

    # (batch_size, num_boxes)
    y_matching_mask = y_true[:, :, -8]

    # tensor of the shape (batch_size)
    # containing # of matching boxes for each image (in the batch)
    num_pos = np.sum(y_matching_mask, axis=1).astype(np.int)
    print('num_pos: {}'.format(num_pos))

    loc_loss = _loc_loss(y_true, y_pred, y_matching_mask, num_pos)
    conf_loss = _conf_loss(y_true, y_pred, y_matching_mask, num_pos, num_boxes, num_classes, hard_neg_pos_ratio)

    return np.sum(conf_loss + loc_alpha * loc_loss)

def _loc_loss(y_true, y_pred, y_matching_mask, num_pos):
    # extract loc classes and data
    y_true_loc = y_true[:, :, :4]
    y_pred_loc = y_pred[:, :, :4]

    # (batch_size, nb_boxes, )
    loc_loss = _smooth_l1_loss(y_true_loc, y_pred_loc)

    # (batch_size)
    loc_pos_loss = np.sum(y_matching_mask * loc_loss, axis=1)

    # average
    loc_pos_loss = loc_pos_loss / num_pos

    return loc_pos_loss

def _conf_loss(y_true, y_pred, y_matching_mask, num_pos, num_boxes, num_classes, hard_neg_pos_ratio):
    # define conf indices including background
    conf_start_idx, conf_end_idx = 4, 4 + num_classes

    # conf loss for all PriorBoxes
    # (batch_size, num_boxes)
    full_conf_loss = _softmax_loss(y_true[:, :, conf_start_idx:conf_end_idx], y_pred[:, :, conf_start_idx:conf_end_idx])

    pos_indices_mask = y_matching_mask
    neg_indices_mask = _mine_hard_examples(full_conf_loss, y_true, y_pred, y_matching_mask,
                                    num_pos, num_boxes, hard_neg_pos_ratio)
    conf_pos_loss = np.sum(pos_indices_mask * full_conf_loss, axis=1)
    conf_neg_loss = np.sum(neg_indices_mask * full_conf_loss, axis=1)

    num_neg = np.sum(neg_indices_mask, axis=1).astype(np.int)

    # average
    conf_pos_loss = conf_pos_loss / num_pos
    conf_neg_loss = conf_neg_loss / num_neg

    print('conf_pos_loss: {}'.format(conf_pos_loss))
    print('conf_neg_loss: {}'.format(conf_neg_loss))

    return conf_pos_loss + conf_neg_loss

def _mine_hard_examples(full_conf_loss, y_true, y_pred, y_matching_mask,
                        num_pos, num_boxes, hard_neg_pos_ratio):
    # hard negative mining

    # tensor of the shape (batch_size)
    # containing # of not matching boxes for each image (in the batch)
    # clipped above using hard_neg_pos_ratio
    num_neg = (num_boxes - num_pos)
    num_neg = (np.minimum(hard_neg_pos_ratio * num_pos, num_neg)).astype(np.int)
    print('num_neg: {}'.format(num_neg))

    #(batch_size, num_boxes)
    all_neg_indices = y_matching_mask == 0
    full_conf_neg_loss = full_conf_loss * all_neg_indices
    #print('full_conf_neg_loss: {}'.format(full_conf_neg_loss))

    top_indices = np.argsort(full_conf_neg_loss, axis=1)
    top_indices_mask = np.zeros_like(top_indices)

    # for each batch
    num_batch = np.shape(y_true)[0]
    for b in range(num_batch):
        top_indices_mask[b][top_indices[b, -num_neg[b]:]] = 1

    return top_indices_mask

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