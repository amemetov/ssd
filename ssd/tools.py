import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects

from .data import PascalVoc2012
import ssd.imaging as imaging

# use show_bboxes instead
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

def show_categories(images, predictions, num_classes, cols, conf_threshold=None, figsize=(12, 8), font_size=14):
    if len(images) != len(predictions):
        raise ValueError("The size of 'images' list should be equal to the size of 'predictions' list")

    if conf_threshold is not None and not 0 <= conf_threshold <= 1:
        raise ValueError("'conf_threshold' should be either None or in range [0, 1]")

    nb_images = len(images)
    rows = nb_images // cols

    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for img, confs, (i, ax) in zip(images, predictions, enumerate(axes.flat)):
        if conf_threshold is None:
            cat_id = np.argmax(confs)
            cat_conf = confs[cat_id]

            __show_categories(ax, img, [cat_id], [cat_conf], colors, font_size)

            # ax.imshow(img)
            # text = ax.text(0, 0, label, verticalalignment='top', fontsize=font_size, bbox={'facecolor': color, 'alpha': 0.5})
        else:
            mask = confs > conf_threshold
            thresholded_categories = np.nonzero(mask)[0]
            thresholded_confs = confs[mask]

            # sort in desc order
            sort_indices = np.argsort(thresholded_confs)[::-1]
            thresholded_categories = thresholded_categories[sort_indices]
            thresholded_confs = thresholded_confs[sort_indices]

            __show_categories(ax, img, thresholded_categories, thresholded_confs, colors, font_size)

    plt.tight_layout()

def __show_categories(ax, img, categories, confs, colors, font_size):
    ax.imshow(img)

    text = ''
    for cat, conf in zip(categories, confs):
        label_idx = int(cat - 1)# take background into account
        label = PascalVoc2012.CLASSES[label_idx] + ': ' + "{0:.2f}".format(conf)
        color = colors[label_idx]
        text = text + label + '\n'

    ax.text(0, 0, text, verticalalignment='top', fontsize=font_size, bbox={'facecolor': color, 'alpha': 0.5})

def show_bboxes(images, bboxes, num_classes, cols, figsize=(12, 8)):
    if len(images) != len(bboxes):
        raise ValueError("The size of 'images' list should be equal to the size of 'bboxes' list")

    nb_images = len(images)
    rows = nb_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for img, boxes, (i, ax) in zip(images, bboxes, enumerate(axes.flat)):
        # when just one box is presented - convert it to array
        if boxes.ndim == 1:
            boxes = np.array([boxes])

        img_w, img_h = img.shape[1], img.shape[0]
        colors = plt.cm.hsv(np.linspace(0, 1, len(boxes))).tolist()

        ax.imshow(img)

        for bbox, color in zip(boxes, colors):
            xmin = int(bbox[0] * img_w)
            ymin = int(bbox[1] * img_h)
            xmax = int(bbox[2] * img_w)
            ymax = int(bbox[3] * img_h)

            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            patch = ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))

def show_predictions(img, predictions, num_classes, figsize=(12, 8), ax=None, font_size=14):
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
        text = ax.text(xmin, ymin, label, verticalalignment='top', fontsize=font_size, bbox={'facecolor': color, 'alpha': 0.5})


"""
Plot train/valid loss curves and save plot to the file ./loss_curve.png.
"""
def plot_loss(history, out_file=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    if out_file is not None:
        plt.savefig(out_file)
    #plt.close()
    plt.tight_layout()

def plot_loss_and_acc(history):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    plt_col1 = axes[0]
    _plot_curve(plt_col1, history.history['acc'], history.history['val_acc'], 'Model Accuracy', 'accuracy', 'epoch')

    plt_col2 = axes[1]
    _plot_curve(plt_col2, history.history['loss'], history.history['val_loss'], 'Model Loss', 'loss', 'epoch')

    plt.tight_layout()

def _plot_curve(plt_col, train_values, valid_values, title, ylabel, xlabel):
    plt_col.plot(train_values)
    plt_col.plot(valid_values)
    plt_col.set_title(title)
    plt_col.set_ylabel(ylabel)
    plt_col.set_xlabel(xlabel)
    plt_col.legend(['train', 'valid'], loc='upper left')

def make_prediction_bboxes(model, target_img_size, base_dir, test_files, bbox_codec):
    orig_images, preprocessed_images, y_pred = make_prediction(model, target_img_size, base_dir, test_files)
    result_bboxes = []
    for encoded_bboxes in y_pred:
        bboxes = bbox_codec.decode(encoded_bboxes)
        result_bboxes.append(bboxes)

    return orig_images, preprocessed_images, result_bboxes

def make_prediction(model, target_img_size, base_dir, test_files):
    orig_images = []
    preprocessed_images = []
    for file in test_files:
        img = imaging.load_img(base_dir + file)
        orig_images.append(img)
        preprocessed_images.append(imaging.preprocess_img(img, target_img_size))

    preprocessed_images = np.array(preprocessed_images)
    y_pred = model.predict(preprocessed_images)
    return orig_images, preprocessed_images, y_pred
