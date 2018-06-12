import unittest
import numpy as np
import time

import ssd.prior_box as pb
from ssd.bbox_codec import intersectionOverUnion, BBoxCodec

class BBoxCodecTest(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(0, intersectionOverUnion([0, 0, 10, 10], [10, 10, 20, 20]))

        self.assertEqual(1, intersectionOverUnion([0, 0, 10, 10], [0, 0, 10, 10]))

        self.assertEqual(1./7, intersectionOverUnion([0, 0, 10, 10], [5, 5, 15, 15]))
        self.assertEqual(1./7, intersectionOverUnion([0, 0, 10, 10], [5, -5, 15, 5]))
        self.assertEqual(1./7, intersectionOverUnion([0, 0, 10, 10], [-5, -5, 5, 5]))
        self.assertEqual(1./7, intersectionOverUnion([0, 0, 10, 10], [-5, 5, 5, 15]))

        self.assertEqual(9./25, intersectionOverUnion([0, 0, 10, 10], [2, 2, 8, 8]))
        self.assertEqual(9./25, intersectionOverUnion([2, 2, 8, 8], [0, 0, 10, 10]))

    def test_encode(self):
        num_classes = 2
        config_1x1 = [
            {'layer_width': 1, 'layer_height': 1, 'num_prior': 6,
             'min_size': 276.0, 'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
        ]
        prior_boxes = pb.create_prior_boxes_vect(300, 300, config_1x1, pb.default_prior_variance)
        self.assertEqual(6, len(prior_boxes), 'Expected 6 prior boxes')

        # expected_pbs = np.array([
        #     [0.04,  0.04,  0.96,  0.96,  0.1, 0.1, 0.2, 0.2],
        #     [0,     0,     1,     1,     0.1, 0.1, 0.2, 0.2],
        #     [0,     0.175, 1,     0.825, 0.1, 0.1, 0.2, 0.2],
        #     [0.175, 0,     0.825, 1,     0.1, 0.1, 0.2, 0.2],
        #     [0,     0.234, 1,     0.766, 0.1, 0.1, 0.2, 0.2],
        #     [0.234, 0,     0.766, 1,     0.1, 0.1, 0.2, 0.2]
        # ])

        bbox_codec = BBoxCodec(prior_boxes, num_classes+1)

        # (num_gtb, 4 + num_classes)
        y_orig = np.array([[0, 0.25, 1, 0.75, 1, 0], [0.25, 0, 0.75, 1, 0, 1]])

        # (num_priors, 4 + (1 + num_classes) + 4 + 4),
        y_encoded_actual, _ = bbox_codec.encode(y_orig)

        # (num_priors, [bg_prob, class1_prob, class2_prob])
        y_expected_assigned_classes = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ])

        self.assertTrue(np.allclose(y_expected_assigned_classes, y_encoded_actual[:, 4:4+3]), 'encoded Y is incorrect')

    def test_perf_encode(self):
        num_classes = 2

        prior_boxes = pb.create_prior_boxes_vect(300, 300, pb.default_config, pb.default_prior_variance)
        bbox_codec = BBoxCodec(prior_boxes, num_classes+1)

        # (num_gtb, 4 + num_classes)
        y_orig = np.array([[0, 0.25, 1, 0.75, 1, 0], [0.25, 0, 0.75, 1, 0, 1]])

        start_time = time.perf_counter()
        y_encoded_actual, _ = bbox_codec.encode(y_orig)
        elapsed_time = time.perf_counter() - start_time
        print("Spent time: {}".format(elapsed_time))
