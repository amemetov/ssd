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

        bbox = NormalizedBBox(label, xmin=gt_data[start_idx + 3], ymin=gt_data[start_idx + 4],
            xmax=gt_data[start_idx + 5], ymax=gt_data[start_idx + 6], difficult=difficult)

        if item_id not in all_gt_bboxes:
            all_gt_bboxes[item_id] = []
        all_gt_bboxes[item_id].append(bbox)



class NormalizedBBox(object):
    def __init__(self, label, xmin, ymin, xmax, ymax, difficult):
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

