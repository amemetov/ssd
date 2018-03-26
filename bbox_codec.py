import numpy as np

class BBoxCodec(object):
    def __init__(self, prior_boxes, num_classes, iou_threshold=0.5):
        self.prior_boxes = prior_boxes
        self.num_priors = len(prior_boxes)
        self.num_classes = num_classes
        self.result_num_classes = num_classes + 1
        self.iou_threshold = iou_threshold

    def encode(self, y_orig):
        """Convert Y from origin format to NN expected format

        # Arguments
            y_orig: 2D tensor of shape (num_gtb, 4 + num_classes), num_classes without background.
                y_orig[:, 0:4] - encoded GTB loc (xmin, ymin, xmax, ymax)
                y_orig[:, 4:4+num_classes] - ground truth one-hot-encoding classes

        # Return
            y_result: 2D tensor of shape (num_priors, 4 + result_num_classes + 4 + 4),
                y_result[:, 0:4] - encoded GTB loc (xmin, ymin, xmax, ymax)
                y_result[:, 4:4+result_num_classes] - ground truth one-hot-encoding classes with background
                y_result[:, -8] - {0, 1} is the indicator for matching the current PriorBox to the GTB,
                not all row has GTB, often it is the background
                y_result[:, -7:] - 0 - is necessary only to have shape as y_pred's shape
        """
        num_gtb = y_orig.shape[0]


        # init y_result by zeros
        y_result = np.zeros((self.num_priors, 4 + self.result_num_classes + 8))

        # by default assume that all boxes are background - set probability of background = 1
        y_result[:, 4] = 1.0
        if num_gtb == 0:
            return y_result


        # TODO: Implement using numpy vectorized methods
        for gtb in y_orig:
            pb_idx = self._find_most_overlapped_pb(gtb)
            if pb_idx is not None:
                pb = self.prior_boxes[pb_idx]
                # copy GTB loc
                y_result[pb_idx][:4] = gtb[:4]

                # probability of the background_class is 0
                y_result[pb_idx][4] = 0.0

                # copy probabilities of categories
                y_result[pb_idx][5:-8] = gtb[4]

                # set indicator to point that this PB matched to GTB - is used by SsdLoss
                y_result[pb_idx][-8] = 1

        return y_result

    def _find_most_overlapped_pb(self, gtb):
        """

        # Arguments
            gtb: tensor of shape (4,) containing (xmin, ymin, xmax, ymax)
        """

        # find the most overlapped PB with iou(gtb, pb) > iou_threshold
        max_iou = 0
        most_overlapped_pb = None
        most_overlapped_pb_idx = None
        for pb_idx in range(self.num_priors):
            pb =self.prior_boxes[pb_idx]
            iou = self._iou(gtb, pb)
            if iou > self.iou_threshold and iou > max_iou:
                max_iou = iou
                most_overlapped_pb = pb
                most_overlapped_pb_idx = pb_idx

        return most_overlapped_pb_idx

    def _iou(self, gtb, pb):
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

        iou = inter_area / union_area
        return iou