import unittest
import numpy as np

import prior_box as pb

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

