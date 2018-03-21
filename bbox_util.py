import numpy as np

def encode_y(y_orig, pbs, num_classes):
    """Convert Y from origin format to NN expected format

    # Arguments
        y_orig: 2D tensor of shape (num_gtb, 4 + num_classes), num_classes without background.
            y_orig[:, :, 0:4] - encoded GTB loc (xmin, ymin, xmax, ymax)
            y_orig[:, :, 4:4+num_classes] - ground truth one-hot-encoding classes

        pbs: List of PriorBox

    # Return
        y_result: 2D tensor of shape (num_priors, 4 + result_num_classes + 4 + 4),
            y_result[:, :, 0:4] - encoded GTB loc (xmin, ymin, xmax, ymax)
            y_result[:, :, 4:4+result_num_classes] - ground truth one-hot-encoding classes with background
            y_result[:, :, -8] - {0, 1} is the indicator for matching the current PriorBox to the GTB,
            not all row has GTB, often it is the background
            y_result[:, :, -7:] - 0 - is necessary only to have shape as y_pred's shape
    """
    num_gtb = y_orig.shape[0]
    num_priors = len(pbs)
    result_num_classes = num_classes + 1

    # init y_result by zeros
    y_result = np.zeros((num_priors, 4 + result_num_classes + 8))

    # by default assume that all boxes are background - set probability of background = 1
    y_result[:, 4] = 1.0
    if num_gtb == 0:
        return y_result


    # TODO: Implement
