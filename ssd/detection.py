import numpy as np
from . import imaging
from .bbox_codec import  intersectionOverUnion

class Detection(object):

    def __init__(self, num_classes, target_img_size, bbox_codec,
                 out_nms_threshold=0.45, out_nms_top_k=400, out_nms_eta=1,
                 out_keep_top_k=200, out_conf_threshold=0.01,
                 eval_overlap_threshold=0.5):
        self.num_classes = num_classes
        self.target_img_size = target_img_size
        self.bbox_codec = bbox_codec
        self.out_nms_threshold = out_nms_threshold
        self.out_nms_top_k = out_nms_top_k
        self.out_nms_eta = out_nms_eta
        self.out_keep_top_k = out_keep_top_k
        self.out_conf_threshold = out_conf_threshold
        self.eval_overlap_threshold = eval_overlap_threshold

    def detect_bboxes(self, model, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]

        preprocessed_imgs = np.zeros((len(imgs), self.target_img_size[0], self.target_img_size[1], 3))
        for i in range(len(imgs)):
            preprocessed_imgs[i] = imaging.preprocess_img(imgs[i], self.target_img_size)
        y_pred = model.predict(preprocessed_imgs)
        return imgs, self.out(imgs, y_pred)

    def out(self, imgs, y_pred):
        """Uses Non maximum suppression to produce final predictions using y_pred produces by the NN

        # Arguments
            y_pred: 3D Tensor of shape (num_batches, num_priors, 4 + num_classes + 4 + 4) - the result of NN eval method

        # Return
            predictions: The result list of predictions for each image.
            Each item contains list of np array [class, conf, xmin, ymin, xmax, ymax]
        """

        predictions = []

        # for each image in the batch
        for img, img_y in zip(imgs, y_pred):
            img_w, img_h = img.shape[1], img.shape[0]
            #print('img_size {}, {}'.format(img_w, img_h))

            bboxes = self.bbox_codec.decode(img_y)

            # convert to abs pos
            bboxes[:, 0] = bboxes[:, 0] * img_w
            bboxes[:, 1] = bboxes[:, 1] * img_h
            bboxes[:, 2] = bboxes[:, 2] * img_w
            bboxes[:, 3] = bboxes[:, 3] * img_h

            # the predictions for the current image
            img_predictions = np.zeros((0, 6))

            for c in range(self.num_classes):
                if c == 0:
                    # skip background
                    continue

                # fetch confs for class c
                confs = img_y[:, 4 + c]
                passed_confs_indices_mask = confs > self.out_conf_threshold

                if np.sum(passed_confs_indices_mask) > 0:
                    passed_bboxes = bboxes[passed_confs_indices_mask]
                    passed_confs = confs[passed_confs_indices_mask]

                    top_k_bboxes, top_k_confs = self.__applyNMSFast(passed_bboxes, passed_confs)
                    top_k_classes = np.full(top_k_confs.shape, c)
                    # create array of [class, conf, xmin, ymin, xmax, ymax]
                    result = np.concatenate((top_k_classes.reshape((top_k_classes.shape[0], 1)),
                                             top_k_confs.reshape((top_k_confs.shape[0], 1)),
                                             top_k_bboxes), axis=-1)
                    img_predictions = np.vstack((img_predictions, result))

            # after processing each class - keep out_keep_top_k results per image.
            if self.out_keep_top_k > -1 and len(img_predictions) > self.out_keep_top_k:
                top_k_indices = np.argsort(img_predictions[:, 1])[:self.out_keep_top_k]
                img_predictions = img_predictions[top_k_indices]

            predictions.append(img_predictions)

        return predictions



    # implementation is ported from the origin SSD caffe implementation (file bbox_util.cpp)
    def __applyNMSFast(self, bboxes, confs):
        #  Get top_k scores (with corresponding indices).
        top_k_indices = np.argsort(confs)
        if len(top_k_indices) > self.out_nms_top_k:
            # take last out_nms_top_k indices cause top_k_indices is ascent ordered
            top_k_indices = top_k_indices[-self.out_nms_top_k:]

        top_k_bboxes = bboxes[top_k_indices]
        top_k_confs = confs[top_k_indices]

        # do nms
        adaptive_threshold = self.out_nms_threshold
        result_indices = []
        for idx in range(len(top_k_indices)):
            keep = True
            for k in range(len(result_indices)):
                if keep:
                    kept_idx = result_indices[k]
                    overlap = intersectionOverUnion(top_k_bboxes[idx], top_k_bboxes[kept_idx])
                    keep = overlap <= adaptive_threshold
                else:
                    break

            if keep:
                result_indices.append(idx)

            if keep and self.out_nms_eta < 1 and adaptive_threshold > 0.5:
                adaptive_threshold *= self.out_nms_eta


        return top_k_bboxes[result_indices], top_k_confs[result_indices]