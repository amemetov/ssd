import math
import numpy as np
from scipy.misc.pilutil import imresize
import bbox_codec

import imaging

class DataAugmenter(object):
    def __init__(self, target_image_size, scale_range=(0.3, 1.0),
                 aspect_ratio_range=(0.5, 2.0), jaccard_overlap_range=(0.0, 1.0)):
        self.target_image_size = target_image_size
        self.scale_range = scale_range
        self.aspect_ratio_range = aspect_ratio_range
        self.jaccard_overlap_range = jaccard_overlap_range

    def augment(self, img, y, do_augment=True):
        # work with the copy of y
        y = np.copy(y)

        if do_augment:
            # 1. Randomly Sample
            img, y = self.__randomly_sample_patch(img, y)

        # 2. Resize to fixed size
        img = imresize(img, self.target_image_size).astype('float32')

        if do_augment:
            # 3. Horizontally flip
            img, y = self.__horizontally_flip(img, y)

            # 4. Photo-metric distortions
            img = self.__apply_photo_metric_distortions(img)

        return img, y

    def __randomly_sample_patch(self, img, y):
        if len(y) == 0:
            return img, y

        if not self.__flip_coin():
            return img, y

        orig_w = img.shape[1]
        orig_h = img.shape[0]

        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        aspect_ratio = np.random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])

        # Calc target dimension
        target_area = scale * orig_w * orig_h
        target_w = min(round(math.sqrt(aspect_ratio * target_area)), orig_w)
        target_h = min(round(target_w / aspect_ratio), orig_h)

        # Crop random sample
        # Randomly choose valid top left coordinates
        target_x = int(np.random.uniform(0.0, orig_w - target_w))
        target_y = int(np.random.uniform(0.0, orig_h - target_h))
        cropped_img = img[target_y:target_y+target_h, target_x:target_x+target_w]
        cropped_bboxes = self.__crop_bboxes(y, orig_w, orig_h, (target_x, target_y, target_w, target_h))

        return cropped_img, cropped_bboxes

    def __crop_bboxes(self, orig_bboxes, orig_img_width, orig_img_height, target_patch_bbox):
        cropped_bboxes = []
        # normalize
        patch_xmin = target_patch_bbox[0] / orig_img_width
        patch_ymin = target_patch_bbox[1] / orig_img_height
        patch_w = target_patch_bbox[2] / orig_img_width
        patch_h = target_patch_bbox[3] / orig_img_height
        patch_bbox = (patch_xmin, patch_ymin, patch_xmin + patch_w, patch_ymin + patch_h)

        # Filter out bboxes which are valid w.r.t center coordinates and min_jaccard_overlap
        for bbox in orig_bboxes:
            if self.__is_valid_bbox(patch_bbox, bbox):
                # convert bbox to the target area
                xmin = max(0, (bbox[0] - patch_xmin) / patch_w)
                ymin = max(0, (bbox[1] - patch_ymin) / patch_h)
                xmax = min(1, (bbox[2] - patch_xmin) / patch_w)
                ymax = min(1, (bbox[3] - patch_ymin) / patch_h)
                bbox[:4] = [xmin, ymin, xmax, ymax]
                cropped_bboxes.append(bbox)

        # Make shape as origin's one
        cropped_bboxes = np.asarray(cropped_bboxes).reshape(-1, orig_bboxes.shape[1])
        return cropped_bboxes

    def __is_valid_bbox(self, target_patch_bbox, bbox):
        # bbox has format (xmin, ymin, xmax, ymax)
        # remember bbox are normalized regarding to the origin image size
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])

        # 1. check whether center is in sampled patch
        if target_patch_bbox[0] < cx < target_patch_bbox[2] and target_patch_bbox[1] < cy < target_patch_bbox[3]:

            if self.jaccard_overlap_range[0] == 0 and self.jaccard_overlap_range[1] == 1:
                # don't check jaccard_overlap
                return True

            # 2. check min jaccard overlap
            iou = bbox_codec.intersectionOverUnion(target_patch_bbox, bbox)
            if self.jaccard_overlap_range[0] <= iou <= self.jaccard_overlap_range[1]:
                return True

        return False

    def __horizontally_flip(self, img, y):
        if len(y) == 0:
            return img, y

        if self.__flip_coin():
            # y - rows of [xmin, ymin, xmax, ymax]
            # flip xmin with xmax
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
            return imaging.flip_horiz(img), y
        return img, y

    def __apply_photo_metric_distortions(self, img):
        if self.__flip_coin():
            img = imaging.randomize_brightness(img)

        if self.__flip_coin():
            img = imaging.randomize_contrast(img)

        if self.__flip_coin():
            img = imaging.randomize_hue(img)

        if self.__flip_coin():
            img = imaging.randomize_saturation(img)

        return img

    def __flip_coin(self):
        return True if np.random.random() < 0.5 else False