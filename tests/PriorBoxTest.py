import unittest
import time
import pickle
import numpy as np

import ssd.prior_box as pb

config = [
    {'layer_width': 38, 'layer_height': 38, 'num_prior': 3,
     'min_size':  30.0, 'max_size': None, 'aspect_ratios': [1.0, 2.0, 1/2.0]},

    {'layer_width': 19, 'layer_height': 19, 'num_prior': 6,
     'min_size':  60.0, 'max_size': 114.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},

    {'layer_width': 10, 'layer_height': 10, 'num_prior': 6,
     'min_size': 114.0, 'max_size': 168.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},

    {'layer_width':  5, 'layer_height':  5, 'num_prior': 6,
     'min_size': 168.0, 'max_size': 222.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},

    {'layer_width':  3, 'layer_height':  3, 'num_prior': 6,
     'min_size': 222.0, 'max_size': 276.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},

    {'layer_width':  1, 'layer_height':  1, 'num_prior': 6,
     'min_size': 276.0, 'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
]

class PriorBoxTest(unittest.TestCase):
    def test_1x1(self):
        config_1x1 = [
            {'layer_width': 1, 'layer_height': 1, 'num_prior': 6,
             'min_size': 276.0, 'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
        ]

        prior_boxes_actual = pb.create_prior_boxes_vect(300, 300, config_1x1, pb.default_prior_variance)
        prior_boxes_expected = np.array([
            [0.04, 0.04, 0.96, 0.96, 0.1, 0.1, 0.2, 0.2],
            [0., 0., 1., 1.,         0.1, 0.1, 0.2, 0.2],
            [0., 0.17473088, 1., 0.82526912, 0.1, 0.1, 0.2, 0.2],
            [0.17473088, 0., 0.82526912, 1., 0.1, 0.1, 0.2, 0.2],
            [0., 0.23441888, 1., 0.76558112, 0.1, 0.1, 0.2, 0.2],
            [0.23441888, 0., 0.76558112, 1., 0.1, 0.1, 0.2, 0.2],
        ])

        self.assertTrue(np.allclose(prior_boxes_expected, prior_boxes_actual))

    def test_ssd300_iter(self):
        # Iter Spent time: 0.021272247002343647
        self.__test_ssd300(False)

    def test_ssd300_vect(self):
        # Vect Spent time: 0.00200256600510329
        self.__test_ssd300(True)

    def __test_ssd300(self, use_vect):
        start_time = time.perf_counter()

        if use_vect:
            prior_boxes_actual = pb.create_prior_boxes_vect(300, 300, config, pb.default_prior_variance)
        else:
            prior_boxes_actual = pb.create_prior_boxes_iter(300, 300, config, pb.default_prior_variance)

        end_time = time.perf_counter()
        print("{} Spent time: {}".format('Vect' if use_vect else 'Iter', end_time - start_time))

        prior_boxes_expected = self.__get_expected_prior_boxes()
        self.assertTrue(np.allclose(prior_boxes_expected, prior_boxes_actual), 'PriorBoxes are incorrect')

    def __get_expected_prior_boxes(self):
        return pickle.load(open('../data/prior_boxes_ssd300.pkl', 'rb'))