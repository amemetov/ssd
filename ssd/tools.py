import numpy as np
import cv2
import matplotlib.pyplot as plt

from .data import PascalVoc2012

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


def show_bboxes(img, predictions, num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    plt.figure(figsize=(24, 12))
    plt.imshow(img)
    currentAxis = plt.gca()

    for prediction in predictions:
        # [class, conf, xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = int(prediction[2]), int(prediction[3]), int(prediction[4]), int(prediction[5])

        label_idx = int(prediction[0] - 1)
        label = PascalVoc2012.CLASSES[label_idx]

        color = colors[label_idx]
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, label, bbox={'facecolor': color, 'alpha': 0.5})

    plt.show()


"""
Plot train/valid loss curves and save plot to the file ./loss_curve.png.
"""
def plot_history_curve(history, out_file=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if out_file is not None:
        plt.savefig(out_file)
    plt.close()