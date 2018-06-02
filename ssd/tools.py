import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects

from .data import PascalVoc2012
import ssd.imaging as imaging

def draw_boxes(img, prior_boxes, log=False):
    if log:
        print('Drawing {} boxes'.format(len(prior_boxes)))

    img = np.copy(img)
    img_w, img_h = img.shape[1], img.shape[0]

    colors = plt.cm.hsv(np.linspace(0, 1, len(prior_boxes))).tolist()
    #print('colors: {}'.format(colors))

    for box, idx in zip(prior_boxes, range(0, len(prior_boxes))):
        xmin = int(box[0] * img_w)
        ymin = int(box[1] * img_h)
        xmax = int(box[2] * img_w)
        ymax = int(box[3] * img_h)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[idx], 1)
        if log:
            print('({},{}), ({}, {})'.format(xmin, ymin, xmax, ymax))

    plt.figure(figsize=(24, 12))
    plt.imshow(img)


def show_bboxes(img, predictions, num_classes, figsize=(24, 12), ax=None, font_size=14):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    img_w, img_h = img.shape[1], img.shape[0]

    ax.imshow(img)
    ax.set_xticks(np.linspace(0, img_w, 8))
    ax.set_yticks(np.linspace(0, img_h, 8))
    ax.grid()
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    #
    for prediction in predictions:
        # [class, conf, xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = int(prediction[2]), int(prediction[3]), int(prediction[4]), int(prediction[5])
        label_idx = int(prediction[0] - 1)
        label = PascalVoc2012.CLASSES[label_idx] + ': ' + "{0:.2f}".format(prediction[1])
        color = colors[label_idx]
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1

        patch = ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        text = ax.text(xmin, ymin, label, verticalalignment='top', fontsize=font_size,
                       bbox={'facecolor': color, 'alpha': 0.5})

def show_classes(imgs, predictions, num_classes, cols, figsize=(24, 12), ax=None, font_size=14):
    nb_images = len(imgs)
    rows = nb_images // cols

    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    for img, pred, (i, ax) in zip(imgs, predictions, enumerate(axes.flat)):
        predicted_class_id = np.argmax(pred)
        label_idx = int(predicted_class_id - 1)
        prob = pred[predicted_class_id]
        label = PascalVoc2012.CLASSES[label_idx] + ': ' + "{0:.2f}".format(prob)
        color = colors[label_idx]

        ax.imshow(img)
        text = ax.text(0, 0, label, verticalalignment='top', fontsize=font_size, bbox={'facecolor': color, 'alpha': 0.5})

    plt.tight_layout()


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


def plot_history_curve_loss_and_acc(history):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    plt_col1 = axes[0]
    _plot_history_curve(plt_col1, history.history['acc'], history.history['val_acc'], 'model accuracy', 'accuracy',
                       'epoch')

    plt_col2 = axes[1]
    _plot_history_curve(plt_col2, history.history['loss'], history.history['val_loss'], 'model loss', 'loss', 'epoch')

    plt.tight_layout()

def _plot_history_curve(plt_col, train_values, valid_values, title, ylabel, xlabel):
    plt_col.plot(train_values)
    plt_col.plot(valid_values)
    plt_col.set_title(title)
    plt_col.set_ylabel(ylabel)
    plt_col.set_xlabel(xlabel)
    plt_col.legend(['train', 'valid'], loc='upper left')

def make_prediction(model, target_img_size, base_dir, test_files):
    test_images = []
    x_test = []
    for file in test_files:
        test_img = imaging.load_img(base_dir + file)
        test_images.append(test_img)
        x_test.append(imaging.preprocess_img(test_img, target_img_size))

    x_test = np.array(x_test)
    y_pred = model.predict(x_test)
    return test_images, x_test, y_pred
