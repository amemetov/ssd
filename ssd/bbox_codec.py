import numpy as np

def intersectionOverUnion(gtb, pb):
    # find left-top and right-bottom of intersection
    left = max(gtb[0], pb[0])
    top = max(gtb[1], pb[1])
    right = min(gtb[2], pb[2])
    bottom = min(gtb[3], pb[3])

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
    def __init__(self, prior_boxes, num_classes, iou_threshold=0.5,
                 encode_variances=True, use_vect=True):
        self.prior_boxes = prior_boxes
        self.num_priors = len(prior_boxes)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.encode_variances = encode_variances
        self.use_vect = use_vect

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
                y_result[:, -8] - {0, 1} is the indicator for matching the current PriorBox to the GTB,
                not all row has GTB, often it is the background
                y_result[:, -7:] - 0 - is necessary only to have shape as y_pred's shape
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
        # explicitly set that by default no matches between PBs and GTBs
        y_result[:, -8] = 0

        num_gtb = y_orig.shape[0]
        if num_gtb == 0:
            return y_result

        if self.use_vect:
            self.__encode_vect(y_orig, y_result)
            #self.__encode_vect_v1(y_orig, y_result)
        else:
            self.__encode_iter(y_orig, y_result)

        return y_result

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

    def __encode_vect(self, y_orig, y_result):
        # 2D tensor, rows are  pbs(y_result)'s indices, columns gtb(y_orig)'s indices
        iou = self.__iou_full(y_orig)

        # The origin implementation encodes in 2 steps:

        # 1. Bipartite matching.
        self.__match_bipartiate(y_orig, y_result, iou)

        # 2. Get most overlapped for the rest prediction bboxes for MatchType_PER_PREDICTION.
        self.__match_per_prediction(y_orig, y_result, iou)

    def __match_bipartiate(self, y_orig, y_result, iou):
        # 1D tensor containing for each GTB indices of most overlapped PBs
        max_pb_indices = np.argmax(iou, axis=0)
        max_gtb_indices = np.arange(iou.shape[1])
        self.__assign_boxes(y_orig, y_result, max_gtb_indices, max_pb_indices)

        # reset IOUs for assigned PBs to not to use them on the step2
        iou[max_pb_indices, :] = 0

    def __match_per_prediction(self, y_orig, y_result, iou):
        # 1D tensor which contains for each PriorBox indices of most overlapped GTB
        max_gtb_indices = np.argmax(iou, axis=1)
        max_iou = iou[np.arange(iou.shape[0]), max_gtb_indices]
        threshold_mask = max_iou > self.iou_threshold
        pb_indices = np.arange(iou.shape[0])[threshold_mask]
        gtb_indices = max_gtb_indices[threshold_mask]

        self.__assign_boxes(y_orig, y_result, gtb_indices, pb_indices)

    def __assign_boxes(self, y_orig, y_result, gtb_indices, pb_indices):
        assert len(gtb_indices) == len(pb_indices)

        gt_boxes = y_orig[gtb_indices]
        gtb_center = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:4])
        gtb_wh = gt_boxes[:, 2:4] - gt_boxes[:, :2]

        prior_boxes = self.prior_boxes[pb_indices]
        pb_center = 0.5 * (prior_boxes[:, :2] + prior_boxes[:, 2:4])
        pb_wh = prior_boxes[:, 2:4] - prior_boxes[:, :2]
        pb_variances = prior_boxes[:, -4:]

        # encode offsets (cx, cy, w, h) relative to the PriorBox coordinates
        # see loss computing in the origin SSD paper
        y_result[pb_indices, :2] = (gtb_center - pb_center) / pb_wh
        y_result[pb_indices, 2:4] = np.log(gtb_wh / pb_wh)

        if self.encode_variances:
            # encode variance of cx, cy, w, h
            y_result[pb_indices, :4] /= pb_variances

        # probability of the background_class is 0
        y_result[pb_indices, 4] = 0.0

        # copy probabilities of categories
        y_result[pb_indices, 5:-8] = gt_boxes[:, 4:]

        # set indicator to point that this PBs matched to GTBs - is used by SsdLoss
        y_result[pb_indices, -8] = 1

    def __encode_vect_v1(self, y_orig, y_result):
        # max_iou - 2D tensor (num_gtb, 2) where
        # max_iou[:, 0] - idx of PB, max_iou[:, 1] - iou value itself
        max_iou = np.apply_along_axis(self.__find_most_overlapped_pb_vect, 1, y_orig)
        threshold_mask = max_iou[:, 1] > self.iou_threshold
        pb_indices = max_iou[threshold_mask, 0].astype(np.int)

        gt_boxes = y_orig[threshold_mask]
        box_center = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:4])
        box_wh = gt_boxes[:, 2:4] - gt_boxes[:, :2]

        prior_boxes = self.prior_boxes[pb_indices]
        pb_center = 0.5 * (prior_boxes[:, :2] + prior_boxes[:, 2:4])
        pb_wh = prior_boxes[:, 2:4] - prior_boxes[:, :2]

        # see loss computing in the origin SSD paper
        y_result[pb_indices, :2] = (box_center - pb_center) / pb_wh
        y_result[pb_indices, 2:4] = np.log(box_wh / pb_wh)

        if self.encode_variances:
            # encode variance of cx, cy
            y_result[pb_indices, :2] /= prior_boxes[:, -4:-2]
            # encode variance of w, h
            y_result[pb_indices, 2:4] /= prior_boxes[:, -2:]

        # probability of the background_class is 0
        y_result[pb_indices, 4] = 0.0

        # copy probabilities of categories
        y_result[pb_indices, 5:-8] = gt_boxes[:, 4:]

        # set indicator to point that this PB matched to GTB - is used by SsdLoss
        y_result[pb_indices, -8] = 1

    def __encode_iter(self, y_orig, y_result):
        for gtb in y_orig:
            pb_idx = self.__find_most_overlapped_pb_iter(gtb)
            if pb_idx is not None:
                box = gtb[:4]
                box_center = 0.5 * (box[:2] + box[2:4])
                box_wh = box[2:4] - box[:2]

                pb = self.prior_boxes[pb_idx]
                pb_center = 0.5 * (pb[:2] + pb[2:4])
                pb_wh = pb[2:4] - pb[:2]

                # see loss computing in the origin SSD paper
                y_result[pb_idx][:2] = (box_center - pb_center) / pb_wh
                y_result[pb_idx][2:4] = np.log(box_wh / pb_wh)

                if self.encode_variances:
                    # encode variance of cx, cy
                    y_result[pb_idx][:2] /= pb[-4:-2]
                    # encode variance of w, h
                    y_result[pb_idx][2:4] /= pb[-2:]

                # probability of the background_class is 0
                y_result[pb_idx][4] = 0.0

                # copy probabilities of categories
                y_result[pb_idx][5:-8] = gtb[4:]

                # set indicator to point that this PB matched to GTB - is used by SsdLoss
                y_result[pb_idx][-8] = 1

    def __find_most_overlapped_pb_vect(self, gtb):
        # calc iou with each prior_boxes in one step
        iou = self.__iou_vect(gtb)
        max_idx = np.argmax(iou)
        return max_idx, iou[max_idx]

    def __iou_full(self, gtbs):
        # add new axis for prior_boxes to manage broadcasting
        pbs = self.prior_boxes[:, np.newaxis]

        # find left-top and right-bottom of intersection
        left = np.maximum(gtbs[:, 0], pbs[:, :, 0])
        top = np.maximum(gtbs[:, 1], pbs[:, :, 1])
        right = np.minimum(gtbs[:, 2], pbs[:, :, 2])
        bottom = np.minimum(gtbs[:, 3], pbs[:, :, 3])

        # area of intersection
        inter_area = (right - left) * (bottom - top)

        gtb_area = (gtbs[:, 2] - gtbs[:, 0]) * (gtbs[:, 3] - gtbs[:, 1])
        # TODO: move calculating pb_area to __init and reuse it
        pb_area = (pbs[:, :, 2] - pbs[:, :, 0]) * (pbs[:, :, 3] - pbs[:, :, 1])

        union_area = gtb_area + pb_area - inter_area

        # 2D tensor, rows are  pbs(y_result)'s indices, columns gtb(y_orig)'s indices
        iou = np.divide(inter_area, union_area, out=np.zeros_like(union_area), where=union_area != 0)
        return iou

    def __iou_vect(self, gtb):
        # find left-top and right-bottom of intersection
        left = np.maximum(gtb[0], self.prior_boxes[:, 0])
        top = np.maximum(gtb[1], self.prior_boxes[:, 1])
        right = np.minimum(gtb[2], self.prior_boxes[:, 2])
        bottom = np.minimum(gtb[3], self.prior_boxes[:, 3])

        # area of intersection
        inter_area = (right - left) * (bottom - top)

        gtb_area = (gtb[2] - gtb[0]) * (gtb[3] - gtb[1])
        # TODO: move calculating pb_area to __init and reuse it
        pb_area = (self.prior_boxes[:, 2] - self.prior_boxes[:, 0]) * (self.prior_boxes[:, 3] - self.prior_boxes[:, 1])

        union_area = gtb_area + pb_area - inter_area

        iou = np.divide(inter_area, union_area, out=np.zeros_like(union_area), where=union_area != 0)
        return iou

    def __find_most_overlapped_pb_iter(self, gtb):
        """

        # Arguments
            gtb: tensor of shape (4,) containing (xmin, ymin, xmax, ymax)
        """

        # find the most overlapped PB with iou(gtb, pb) > iou_threshold
        max_iou = 0
        most_overlapped_pb_idx = None
        for pb_idx in range(self.num_priors):
            pb =self.prior_boxes[pb_idx]
            iou = intersectionOverUnion(gtb, pb)
            if iou > self.iou_threshold and iou > max_iou:
                max_iou = iou
                most_overlapped_pb_idx = pb_idx

        return most_overlapped_pb_idx

