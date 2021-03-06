import numpy as np

def intersectionOverUnion(gtb, pb):
    # find left-top and right-bottom of intersection
    left = max(gtb[0], pb[0])
    top = max(gtb[1], pb[1])
    right = min(gtb[2], pb[2])
    bottom = min(gtb[3], pb[3])

    right = np.maximum(right, left)
    bottom = np.maximum(bottom, top)

    # area of intersection
    inter_area = (right - left) * (bottom - top)

    gtb_area = (gtb[2] - gtb[0]) * (gtb[3] - gtb[1])
    pb_area = (pb[2] - pb[0]) * (pb[3] - pb[1])

    union_area = gtb_area + pb_area - inter_area

    if union_area == 0:
        # avoid divide by zero
        return 0

    iou = inter_area / union_area
    return iou

class BBoxCodec(object):
    def __init__(self, prior_boxes, num_classes,
                 iou_threshold=0.5, encode_variances=True, match_per_prediction=True):
        self.prior_boxes = prior_boxes
        self.num_priors = len(prior_boxes)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.encode_variances = encode_variances
        self.match_per_prediction = match_per_prediction

        # add new axis for prior_boxes to manage broadcasting
        self.pb_broadcasted = self.prior_boxes[:, np.newaxis]
        self.pb_area = (self.pb_broadcasted[:, :, 2] - self.pb_broadcasted[:, :, 0]) * (self.pb_broadcasted[:, :, 3] - self.pb_broadcasted[:, :, 1])

    def encode(self, y_orig):
        """Convert Y from origin format to NN expected format

        # Arguments
            y_orig: 2D tensor of shape (num_gtb, 4 + num_classes), num_classes without background.
                y_orig[:, 0:4] - GTB loc (xmin, ymin, xmax, ymax) normalized by corresponding image size
                y_orig[:, 4:4+num_classes] - ground truth one-hot-encoding classes

        # Return
            y_result: 2D tensor of shape (num_priors, 4 + num_classes + 4 + 4),
                y_result[:, 0:4] - encoded GTB offsets in bbox locations (cx, cy, w, h)
                y_result[:, 4:4+num_classes] - ground truth one-hot-encoding classes with background
                y_result[:, -8:] - 0 - is necessary only to have shape as y_pred's shape
        """

        # filter out empty boxes
        gt_boxes = y_orig
        gtb_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gtb_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        gtb_area = gtb_w * gtb_h
        y_orig = y_orig[gtb_area > 0]

        # init y_result by zeros
        y_result = np.zeros((self.num_priors, 4 + self.num_classes + 8))

        # by default assume that all boxes are background - set probability of background = 1
        y_result[:, 4] = 1.0

        # for each PB idx set assigned GTB idx
        assign_result = np.full((self.num_priors), -1).astype(np.int)

        num_gtb = y_orig.shape[0]
        if num_gtb == 0:
            return y_result, assign_result

        self.__encode(y_orig, y_result, assign_result)

        return y_result, assign_result

    def decode(self, y_pred):
        """ Convert Y predicted into bboxes.

        # Arguments
            y_pred: 2D tensor of shape (num_priors, 4 + num_classes + 4 + 4),
                y_pred[:, 0:4] - predicted offsets in bbox locations (cx, cy, w, h) - see encode method
                y_pred[:, 4:4+num_classes] - predicted one-hot-encoding classes with background
                y_pred[:, -8:-4] - PriorBox loc
                y_pred[:, -4:] - PriorBox variances
        # Return
            bboxes: 2D tensor of shape (num_priors, 4),
                bboxes[:, 0:4] - (xmin, ymin, xmax, ymax)
        """
        num_priors = y_pred.shape[0]

        # init bboxes by zeros
        bboxes  = np.zeros((num_priors, 4))

        # (cx, cy, w, h)
        encoded_loc = y_pred[:, 0:4]
        pb_loc = y_pred[:, -8:-4]
        pb_variances = y_pred[:, -4:]

        pb_center = 0.5 * (pb_loc[:, :2] + pb_loc[:, 2:4])
        pb_wh = pb_loc[:, 2:4] - pb_loc[:, :2]

        # decode bbox center and size - see encode method
        if self.encode_variances:
            decode_box_center = encoded_loc[:, :2] * pb_wh * pb_variances[:, :2] + pb_center
            decode_box_wh = np.exp(encoded_loc[:, 2:4] * pb_variances[:, 2:4]) * pb_wh
        else:
            decode_box_center = encoded_loc[:, :2] * pb_wh + pb_center
            decode_box_wh = np.exp(encoded_loc[:, 2:4]) * pb_wh

        # decode bbox loc from box center and size
        bboxes[:, :2] = decode_box_center - 0.5 * decode_box_wh
        bboxes[:, 2:4] = decode_box_center + 0.5 * decode_box_wh

        bboxes = np.clip(bboxes, 0, 1)

        return bboxes

    def __encode(self, y_orig, y_result, assign_result):
        # 2D tensor, rows are  pbs(y_result)'s indices, columns gtb(y_orig)'s indices
        iou = self.__iou_full(y_orig)

        # The origin implementation encodes in 2 steps:

        # 1. Bipartite matching.
        self.__match_bipartiate(y_orig, y_result, assign_result, iou)

        # 2. Get most overlapped for the rest prediction bboxes for MatchType_PER_PREDICTION.
        if self.match_per_prediction:
            self.__match_per_prediction(y_orig, y_result, assign_result, iou)

    def __match_bipartiate(self, y_orig, y_result, assign_result, iou, gtb_indices=None):
        # pb_indices - 1D tensor containing for each GTB indices of most overlapped PBs

        if gtb_indices is None:
            gtb_indices = np.arange(iou.shape[1])
            pb_indices = np.argmax(iou, axis=0)
            max_iou = np.max(iou, axis=0)
        else:
            iou_curr = iou[:, gtb_indices]
            pb_indices = np.argmax(iou_curr, axis=0)
            max_iou = np.max(iou_curr, axis=0)

        if len(gtb_indices) == 0:
            return

        unique_pb_indices = np.unique(pb_indices)
        if pb_indices.shape[0] == unique_pb_indices.shape[0]:
            self.__assign_boxes(y_orig, y_result, assign_result, gtb_indices, pb_indices, iou)
        else:
            # solve possible duplications in pb_indices
            # start using from the most overlapped pairs
            max_iou_sort_args = np.argsort(max_iou)[::-1]
            ordered_gtb_indices = gtb_indices[max_iou_sort_args]
            ordered_pb_indices = pb_indices[max_iou_sort_args]

            # get only first appeared indices
            ordered_unique_pb_indices, unique_indices = np.unique(ordered_pb_indices, return_index=True)
            max_gtb_indices = ordered_gtb_indices[unique_indices]
            max_pb_indices = ordered_pb_indices[unique_indices]

            # assign most overlapped pairs
            self.__assign_boxes(y_orig, y_result, assign_result, max_gtb_indices, max_pb_indices, iou)

            # process remain GTBs
            remain_gtb_indices = np.setdiff1d(ordered_gtb_indices, max_gtb_indices, assume_unique=True)
            self.__match_bipartiate(y_orig, y_result, assign_result, iou, remain_gtb_indices)

    def __match_per_prediction(self, y_orig, y_result, assign_result, iou):
        # 1D tensor which contains for each PriorBox indices of most overlapped GTB
        max_gtb_indices = np.argmax(iou, axis=1)
        max_iou = iou[np.arange(iou.shape[0]), max_gtb_indices]
        threshold_mask = max_iou > self.iou_threshold
        pb_indices = np.arange(iou.shape[0])[threshold_mask]
        gtb_indices = max_gtb_indices[threshold_mask]

        self.__assign_boxes(y_orig, y_result, assign_result, gtb_indices, pb_indices, iou)

    def __assign_boxes(self, y_orig, y_result, assign_result, gtb_indices, pb_indices, iou_full):
        assert len(gtb_indices) == len(pb_indices)
        if len(gtb_indices) == 0:
            return

        # store overlap mask before resetting
        overlap_mask = iou_full[pb_indices, gtb_indices] > 0
        #overlap_mask = iou_full[pb_indices, gtb_indices] > (self.iou_threshold / 2)

        # mark PBs as assigned to GTBs
        iou_full[pb_indices, :] = -1

        # print('------------------------------')
        # print('pb_indices: {}'.format(pb_indices))
        # print('gtb_indices: {}'.format(gtb_indices))

        # use only pairs which have overlap
        pb_indices = pb_indices[overlap_mask]
        gtb_indices = gtb_indices[overlap_mask]

        # store matched indices
        # print('pb_indices: {}'.format(pb_indices))
        # print('gtb_indices: {}'.format(gtb_indices))
        assign_result[pb_indices] = gtb_indices

        # get GTBs
        gt_boxes = y_orig[gtb_indices]
        gtb_center = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:4])
        gtb_wh = gt_boxes[:, 2:4] - gt_boxes[:, :2]

        # get PBs
        prior_boxes = self.prior_boxes[pb_indices]
        pb_center = 0.5 * (prior_boxes[:, :2] + prior_boxes[:, 2:4])
        pb_wh = prior_boxes[:, 2:4] - prior_boxes[:, :2]


        # encode offsets (cx, cy, w, h) relative to the PriorBox coordinates
        # see loss computing in the origin SSD paper
        y_result[pb_indices, :2] = (gtb_center - pb_center) / pb_wh
        y_result[pb_indices, 2:4] = np.log(gtb_wh / pb_wh)

        if self.encode_variances:
            # encode variance of cx, cy, w, h
            pb_variances = prior_boxes[:, -4:]
            y_result[pb_indices, :4] /= pb_variances

        # copy probabilities of categories (background is included)
        y_result[pb_indices, 4:-8] = gt_boxes[:, 4:]

    def __iou_full(self, gtbs):
        # find left-top and right-bottom of intersection
        left = np.maximum(gtbs[:, 0], self.pb_broadcasted[:, :, 0])
        top = np.maximum(gtbs[:, 1], self.pb_broadcasted[:, :, 1])
        right = np.minimum(gtbs[:, 2], self.pb_broadcasted[:, :, 2])
        bottom = np.minimum(gtbs[:, 3], self.pb_broadcasted[:, :, 3])

        right = np.maximum(right, left)
        bottom = np.maximum(bottom, top)

        # area of intersection
        inter_area = (right - left) * (bottom - top)

        gtb_area = (gtbs[:, 2] - gtbs[:, 0]) * (gtbs[:, 3] - gtbs[:, 1])

        union_area = gtb_area + self.pb_area - inter_area

        # 2D tensor, rows are  pbs(y_result)'s indices, columns gtb(y_orig)'s indices
        iou = np.divide(inter_area, union_area, out=np.zeros_like(union_area), where=union_area != 0)
        return iou


class LargestObjClassCodec(object):
    def encode(self, y_orig):
        #print('y_orig: {}'.format(y_orig))
        # encode - take the largest bbox
        # y_orig [:, 0:4] - GTB loc (xmin, ymin, xmax, ymax) normalized by corresponding image size
        y_encoded = sorted(y_orig, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)[0]
        # keep only one-hot-encoded classes + background at the beginning
        y_encoded = np.concatenate(([0], y_encoded[4:]))

        # for each PB idx set assigned GTB idx
        assign_result = None  # np.full((self.num_priors), -1).astype(np.int)

        return y_encoded, assign_result

class LargestObjBoxCodec(object):
    def __init__(self, code_type='center_size'):
        if code_type not in ['center_size', 'coords']:
            raise ValueError("code_type should be on of ['center_size', 'coords']")

        self.code_type = code_type

    def encode(self, y_orig):
        # encode - take the largest bbox
        # y_orig [:, 0:4] - GTB loc (xmin, ymin, xmax, ymax) normalized by corresponding image size
        y_encoded = sorted(y_orig, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)[0]

        bbox = y_encoded[:4]

        if self.code_type == 'center_size':
            # convert to center with size
            bbox_center = 0.5 * (bbox[:2] + bbox[2:4])
            bbox_wh = bbox[2:4] - bbox[:2]
            bbox = np.concatenate((bbox_center, bbox_wh))

        # keep only bbox
        y_encoded = bbox

        # for each PB idx set assigned GTB idx
        assign_result = None  # np.full((self.num_priors), -1).astype(np.int)

        return y_encoded, assign_result

    def decode(self, y_pred):
        bbox = y_pred#[0:4]

        if self.code_type == 'center_size':
            bbox_center = bbox[:2]
            bbox_wh = bbox[2:]
            # decode bbox loc from box center and size
            bbox[:2] = bbox_center - 0.5 * bbox_wh
            bbox[2:] = bbox_center + 0.5 * bbox_wh

        bbox = np.clip(bbox, 0, 1)
        return bbox

class LargestObjBoxAndClassCodec(object):
    def __init__(self, code_type='center_size'):
        if code_type not in ['center_size', 'coords']:
            raise ValueError("code_type should be on of ['center_size', 'coords']")

        self.code_type = code_type

    def encode(self, y_orig):
        # encode - take the largest bbox
        # y_orig [:, 0:4] - GTB loc (xmin, ymin, xmax, ymax) normalized by corresponding image size
        y_encoded = sorted(y_orig, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)[0]

        if self.code_type == 'center_size':
            # convert to center with size
            bbox_center = 0.5 * (y_encoded[:2] + y_encoded[2:4])
            bbox_wh = y_encoded[2:4] - y_encoded[:2]
            y_encoded[:2] = bbox_center
            y_encoded[2:4] = bbox_wh

        y_encoded = np.concatenate((y_encoded[:4], [0], y_encoded[4:]))

        # for each PB idx set assigned GTB idx
        assign_result = None  # np.full((self.num_priors), -1).astype(np.int)

        return y_encoded, assign_result

    def decode(self, y_pred):
        if self.code_type == 'center_size':
            bbox_center = y_pred[:2]
            bbox_wh = y_pred[2:]
            # decode bbox loc from box center and size
            y_pred[:2] = bbox_center - 0.5 * bbox_wh
            y_pred[2:4] = bbox_center + 0.5 * bbox_wh

        return y_pred

