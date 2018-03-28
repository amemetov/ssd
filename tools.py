import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_boxes(img, prior_boxes, log=False):
    if log:
        print('Drawing {} boxes'.format(len(prior_boxes)))

    img = np.copy(img)
    img_w, img_h = img.shape[1], img.shape[0]

    for box, idx in zip(prior_boxes, range(0, len(prior_boxes))):
        xmin = int(box[0] * img_w)
        ymin = int(box[1] * img_h)
        xmax = int(box[2] * img_w)
        ymax = int(box[3] * img_h)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), ((idx + 1) / (len(prior_boxes)) * 255, 0, 0), 1)
        if log:
            print('({},{}), ({}, {})'.format(xmin, ymin, xmax, ymax))

    plt.figure(figsize=(24, 12))
    plt.imshow(img)