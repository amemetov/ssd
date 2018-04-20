import unittest
import numpy as np
import time

import keras.backend as K
import tensorflow as tf

from keras_impl.losses import SsdLoss
import np_losses

class LossTest(unittest.TestCase):
    def test_loss(self):

        num_classes = 2
        y_true = np.array([
            # image 1
            [
                # xmin, ymin, xmax, ymax, bg, cl1, cl2, gtb_mask, 7 zeros
                [0.25, 0.25, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0.75, 0.75, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0.75, 0.75, 1.0, 1.0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
            ],
            # image 2
            [
                # xmin, ymin, xmax, ymax, bg, cl1, cl2, gtb_mask, 7 zeros
                [0.0, 0.0, 0.25, 0.25, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.25, 0.25, 0.5, 0.5, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0.75, 0.75, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
            ]
        ])

        y_pred = np.array([
            # image 1
            [
                # xmin, ymin, xmax, ymax, bg, cl1, cl2, gtb_mask, 7 zeros
                [0.0, 0.25, 0.5, 0.5, 0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0.75, 0.75, 0, 0.6, 0.4, 1, 0, 0, 0, 0, 0, 0, 0],
                [0.75, 0.75, 1.0, 1.0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
            ],
            # image 2
            [
                # xmin, ymin, xmax, ymax, bg, cl1, cl2, gtb_mask, 7 zeros
                [0.0, 0.0, 0.25, 0.25, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.25, 0.25, 0.5, 0.5, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0.75, 0.75, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
            ]
        ])


        expected_loss = np_losses.loss(y_true, y_pred, num_classes)
        actual_loss = loss_keras(y_true, y_pred, num_classes)

        self.assertAlmostEqual(expected_loss, actual_loss, delta=0.000001)


def loss_keras(y_true, y_pred, num_classes, loc_alpha=1.0, hard_neg_pos_ratio=3.0, background_class_id=0):
    y_true_keras = K.variable(y_true)
    y_pred_keras = K.variable(y_pred)
    loss = SsdLoss(num_classes=num_classes)
    loss_val = loss.loss(y_true_keras, y_pred_keras)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        result = loss_val.eval()
        return result


