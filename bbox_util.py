"""
# References
    - https://github.com/weiliu89/caffe/blob/ssd/include/caffe/util/bbox_util.hpp
    - https://github.com/weiliu89/caffe/blob/ssd/src/caffe/util/bbox_util.cpp
"""


def GetGroundTruth(gt_data, num_gt, background_label_id, use_difficult_gt, all_gt_bboxes):
    """Retrieve bounding box ground truth from gt_data.

    # Arguments
        gt_data: list of 1 x 1 x num_gt x 7 blob.
        num_gt: the number of ground truth.
        background_label_id: the label for background class which is used to do
            santity check so that no ground truth contains it.
        use_difficult_gt: boolean
        all_gt_bboxes: dictionary <int, [NormalizedBBox]>
            stores ground truth for each image.
            Label of each bbox is stored in NormalizedBBox.
    """

    #del all_gt_bboxes[:]
    all_gt_bboxes.clear()

    for i in range(0, num_gt):
        start_idx = i * 8
        item_id = gt_data[start_idx]
        if item_id == -1:
            continue

        label = gt_data[start_idx + 1]
        if background_label_id == label:
            raise ValueError("Found background label in the dataset.")

        # boolean
        difficult = gt_data[start_idx + 7]
        if not use_difficult_gt and difficult:
            #Skip reading difficult ground truth.
            continue

        bbox = NormalizedBBox(label=label, xmin=gt_data[start_idx + 3], ymin=gt_data[start_idx + 4],
                              xmax=gt_data[start_idx + 5], ymax=gt_data[start_idx + 6], difficult=difficult)

        if item_id not in all_gt_bboxes:
            all_gt_bboxes[item_id] = []
        all_gt_bboxes[item_id].append(bbox)


def GetPriorBBoxes(prior_data, num_priors, prior_bboxes, prior_variances):
    """Get prior bounding boxes from prior_data.

    # Arguments
        prior_data: 1 x 2 x num_priors * 4 x 1 blob.
        num_priors: number of priors.
        prior_bboxes: stores all the prior bboxes in the format of NormalizedBBox.
        prior_variances: stores all the variances needed by prior bboxes.
    """
    del prior_bboxes[:]
    del prior_variances[:]

    for i in range(0, num_priors):
        start_idx = i * 4
        bbox = NormalizedBBox(label=None, xmin=prior_data[start_idx], ymin=prior_data[start_idx + 1],
                              xmax=prior_data[start_idx + 2], ymax=prior_data[start_idx + 3], difficult=None)
        prior_bboxes.append(bbox)

    for i in range(0, num_priors):
        start_idx = (num_priors + i) * 4
        # var = []
        # for j in range(0, 4):
        #     var.append(prior_data[start_idx + j])
        var = prior_data[start_idx:start_idx+4]
        prior_variances.append(var)


def GetLocPredictions(loc_data, num, num_preds_per_class, num_loc_classes,
                      share_location, loc_preds):
    """Get location predictions from loc_data.

    # Arguments
        loc_data: num x num_preds_per_class * num_loc_classes * 4 blob.
        num: the number of images.
        num_preds_per_class: number of predictions per class.
        num_loc_classes: number of location classes.
            It is 1 if share_location is true;
            and is equal to number of classes needed to predict otherwise.
        share_location: if true, all classes share the same location prediction.
        loc_preds: stores the location prediction, where each item contains location prediction for an image.
            vector<LabelBBox>* loc_preds where LabelBBox is:
            typedef map<int, vector<NormalizedBBox> > LabelBBox;
    """
    del loc_preds[:]

    if share_location and num_loc_classes != 1:
        raise ValueError("When share_location is True, num_loc_classes should be 1, but got {}".format(num_loc_classes))

    loc_data_start_idx = 0
    for i in range(0, num):

        label_bbox = {}
        loc_preds.append(label_bbox)
        for p in range(0, num_preds_per_class):
            start_idx = loc_data_start_idx + p * num_loc_classes * 4

            for c in range(0, num_loc_classes):
                label = -1 if share_location else c

                if label not in label_bbox:
                    label_bbox[label] = [NormalizedBBox(label=label)]*num_preds_per_class

                label_bbox[label][p].xmin = loc_data[start_idx + c * 4]
                label_bbox[label][p].ymin = loc_data[start_idx + c * 4 + 1]
                label_bbox[label][p].xmax = loc_data[start_idx + c * 4 + 2]
                label_bbox[label][p].ymax = loc_data[start_idx + c * 4 + 3]

        loc_data_start_idx += num_preds_per_class * num_loc_classes * 4

class NormalizedBBox(object):
    def __init__(self, label=None, xmin=None, ymin=None, xmax=None, ymax=None, difficult=None):
        self._label = label
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax
        self._difficult = difficult
        self._size = None
        self._size = BBoxSize(self)

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, value):
        self._xmin = value

    @property
    def ymin(self):
        return self._ymin

    @ymin.setter
    def ymin(self, value):
        self._ymin = value

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, value):
        self._xmax = value

    @property
    def ymax(self):
        return self._ymax

    @ymax.setter
    def ymax(self, value):
        self._ymax = value

    def has_size(self):
        return self._size is not None

    @property
    def size(self):
        return self._size


# Compute bbox size.
def BBoxSize(bbox, normalized=True):
    if bbox.xmax is None or bbox.xmin is None or bbox.ymax is None or bbox.ymin is None:
        return None

    if bbox.xmax < bbox.xmin or bbox.ymax < bbox.ymin:
        #If bbox is invalid (e.g.xmax < xmin or ymax < ymin), return 0.
        return 0
    else:
        if bbox.has_size():
            return bbox.size
        else:
            width = bbox.xmax - bbox.xmin
            height = bbox.ymax - bbox.ymin
            if normalized:
                return width * height
            else:
                # If bbox is not within range[0, 1].
                return (width + 1) * (height + 1)

