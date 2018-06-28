import os
import numpy as np

from . import imaging
from .data import PascalVoc2012

class PascalVocEval(object):
    def __init__(self, gtb, img_dir, classes = PascalVoc2012.CLASSES,
                 use_07_metric=False, iou_threshold=0.5, batch_size=16):
        self.gtb = gtb
        self.img_dir = img_dir
        self.use_07_metric = use_07_metric
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        self.classes = classes

    def eval(self, model, detector, test_files, return_per_class=False):
        # make predictions
        predictions = []
        img_dims = []
        for offset in range(0, len(test_files), self.batch_size):
            samples_batch = test_files[offset:offset + self.batch_size]
            images_batch = []

            for img_file_name in samples_batch:
                img_full_path = os.path.join(self.img_dir, img_file_name)
                img = imaging.load_img(img_full_path).astype(np.float32)
                images_batch.append(img)
                img_dims.append(img.shape)

            _, predictions_batch = detector.detect_bboxes(model, images_batch)
            predictions = predictions + predictions_batch

        recalls, precisions, aps = self._do_eval(test_files, img_dims, predictions)
        if return_per_class:
            return recalls, precisions, aps
        else:
            return np.mean(aps)

    def _do_eval(self, img_names, img_dims, predictions):
        if len(img_names) != len(predictions):
            raise ValueError("Number of 'images' must be equal to number of 'predictions'")

        recalls = []
        precisions = []
        aps = []
        for c_idx, c_name in enumerate(self.classes):
            class_recalls, class_precisions, class_aps = self._voc_eval(img_names, img_dims, predictions, c_idx, c_name)
            recalls.append(class_recalls)
            precisions.append(class_precisions)
            aps.append(class_aps)

        return recalls, precisions, aps

    def _voc_eval(self, img_names, img_dims, predictions, class_idx, class_name):
        # extract gt objects for this class
        class_gt = {}

        # total # of GTBs (excluding difficult ones)
        num_total_gtbs = 0

        for img_name, img_dim in zip(img_names, img_dims):
            # find all GTs for class_idx
            img_gtbs = self.gtb[img_name]
            # filter out GTs only for passed class_name
            class_gts = [obj for obj in img_gtbs if obj[4 + class_idx] == 1]
            bbox = np.array([x[:4] for x in class_gts])
            difficult = np.array([x[-1] for x in class_gts]).astype(np.bool)
            # to track already matched GTBs
            matched_gtbs = [False] * len(class_gts)
            num_total_gtbs = num_total_gtbs + sum(~difficult)
            class_gt[img_name] = {'bbox': bbox, 'difficult': difficult, 'matched_gtbs': matched_gtbs}

        # Prepare predictions
        image_names = [] # image_name for each Predicted Box
        pred_confs = []
        pred_bbs = []
        pred_classes = []

        for img_name, img_dim, img_preds in zip(img_names, img_dims, predictions):
            img_w, img_h = img_dim[1], img_dim[0]

            # img_preds is a list of [class, conf, xmin, ymin, xmax, ymax]
            for pred in img_preds:
                image_names.append(img_name)
                pred_classes.append(pred[0])
                pred_confs.append(pred[1])

                bbox = pred[2:]
                # convert absolute coords to relative coords
                bbox[0] /= img_w
                bbox[1] /= img_h
                bbox[2] /= img_w
                bbox[3] /= img_h
                pred_bbs.append(bbox)

        # convert to np array
        pred_confs = np.array(pred_confs)
        pred_bbs = np.array(pred_bbs)

        # sort by confidence
        sorted_ind = np.argsort(-pred_confs)
        sorted_scores = np.sort(-pred_confs)
        pred_bbs = pred_bbs[sorted_ind, :]
        image_names = [image_names[x] for x in sorted_ind]

        # total # of Predicted Boxes
        num_preds = len(image_names)
        tp = np.zeros(num_preds) # True Positives
        fp = np.zeros(num_preds) # False Positives
        for d in range(num_preds):
            img_class_gt = class_gt[image_names[d]]
            pred_bbox = pred_bbs[d, :].astype(float)
            max_overlap = -np.inf
            img_gtbs = img_class_gt['bbox'].astype(float)

            if img_gtbs.size > 0:
                overlaps = self._iou(img_gtbs, pred_bbox)
                max_overlap = np.max(overlaps)
                max_overlap_idx = np.argmax(overlaps)

            if max_overlap > self.iou_threshold:
                if not img_class_gt['difficult'][max_overlap_idx]:
                    if not img_class_gt['matched_gtbs'][max_overlap_idx]:
                        tp[d] = 1.
                        img_class_gt['matched_gtbs'][max_overlap_idx] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        # avoid divide by zero in case no GTB is presented for this class
        recall = tp / np.maximum(float(num_total_gtbs), np.finfo(np.float64).eps)
        # avoid divide by zero in case the first detection matches a difficult ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self._voc_ap(recall, precision)

        return recall, precision, ap

    def _iou(self, img_gtbs, pred_bbox):
        ixmin = np.maximum(img_gtbs[:, 0], pred_bbox[0])
        iymin = np.maximum(img_gtbs[:, 1], pred_bbox[1])
        ixmax = np.minimum(img_gtbs[:, 2], pred_bbox[2])
        iymax = np.minimum(img_gtbs[:, 3], pred_bbox[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inter_area = iw * ih

        bbox_area = (pred_bbox[2] - pred_bbox[0] + 1.) * (pred_bbox[3] - pred_bbox[1] + 1.)
        gtb_area = (img_gtbs[:, 2] - img_gtbs[:, 0] + 1.) * (img_gtbs[:, 3] - img_gtbs[:, 1] + 1.)
        union_area = bbox_area + gtb_area - inter_area

        overlaps = inter_area / union_area
        return overlaps

    def _voc_ap(self, recalls, precisions):
        """
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the VOC 07 11 point method.
        """
        if self.use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], recalls, [1.]))
            mpre = np.concatenate(([0.], precisions, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap