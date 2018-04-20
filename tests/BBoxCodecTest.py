import unittest
import numpy as np
import time

import prior_box as pb
from bbox_codec import intersectionOverUnion, BBoxCodec

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

    def test_encode_iter(self):
        self.__test_encode(False)

    def test_encode_vect(self):
        self.__test_encode(True)

    def test_perf_encode_iter(self):
        # Iter Spent time: 0.07274351599335205
        self.__test_perf_encode(False)

    def test_perf_encode_vect(self):
        # Vect Spent time: 0.0006487050122814253
        self.__test_perf_encode(True)

    def __test_encode(self, use_vect):
        num_classes = 2
        config_1x1 = [
            {'layer_width': 1, 'layer_height': 1, 'num_prior': 6,
             'min_size': 276.0, 'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
        ]
        prior_boxes = pb.create_prior_boxes_vect(300, 300, config_1x1, pb.default_prior_variance)
        self.assertEqual(6, len(prior_boxes), 'Expected 6 prior boxes')

        bbox_codec = BBoxCodec(prior_boxes, num_classes+1, use_vect=use_vect)

        # (num_gtb, 4 + num_classes)
        y_orig = np.array([[0, 0.25, 1, 0.75, 1, 0], [0.25, 0, 0.75, 1, 0, 1]])

        y_encoded_actual = bbox_codec.encode(y_orig)

        # we expect 6 rows
        # (num_priors, 4 + (1 + num_classes) + 4 + 4),
        y_encoded_expected = np.array([
            # (xmin, ymin, xmax, ymax, background_prob, class1, class2, gtb_mark, 7 zeros)
            [0,    0,    0,    0,    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0,    0,    0,    0,    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0,    0,    0,    0,    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0,    0,    0,    0,    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #[0,    0.25, 1,    0.75,    0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            #[0.25, 0,    0.75, 1,       0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

            #
            #[0, 0, 0, -0.0604594272868,  0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            #[0, 0,  -0.0604594272868, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

            # when encoded variances
            [0, 0, 0, -0.30229714,  0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -0.30229714, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

        ])

        self.assertTrue(np.allclose(y_encoded_expected, y_encoded_actual), 'encoded Y is incorrect')


    def __test_perf_encode(self, use_vect):
        num_classes = 2

        prior_boxes = pb.create_prior_boxes_vect(300, 300, pb.default_config, pb.default_prior_variance)
        bbox_codec = BBoxCodec(prior_boxes, num_classes+1, use_vect=use_vect)

        # (num_gtb, 4 + num_classes)
        y_orig = np.array([[0, 0.25, 1, 0.75, 1, 0], [0.25, 0, 0.75, 1, 0, 1]])

        start_time = time.perf_counter()
        y_encoded_actual = bbox_codec.encode(y_orig)
        elapsed_time = time.perf_counter() - start_time
        print("{} Spent time: {}".format('Vect' if use_vect else 'Iter', elapsed_time))
