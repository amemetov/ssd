import prior_box as pb
import pickle

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

if __name__ == "__main__":
    actual_prior_boxes = pb.create_prior_boxes(300, 300, config, pb.default_prior_variance)
    expected_prior_boxes = pickle.load(open('data/prior_boxes_ssd300.pkl', 'rb'))
    diff = actual_prior_boxes - expected_prior_boxes
    print("diff.shape {}, max value {}, min value {}".format(diff.shape, diff.max(), diff.min()))




