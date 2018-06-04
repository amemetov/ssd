import numpy as np
import cv2
import matplotlib.pyplot as plt

from .data import PascalVoc2012
import ssd.imaging as imaging

# use show_bboxes instead
def draw_boxes(img, prior_boxes, log=False):
    if log:
        print('Drawing {} boxes'.format(len(prior_boxes)))

    img = np.copy(img)
    img_w, img_h = img.shape[1], img.shape[0]

    colors = plt.cm.hsv(np.linspace(0, 1, len(prior_boxes))).tolist()

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
        ax.imshow(img)

        if conf_threshold is None:
            cat_id = np.argmax(confs)
            cat_conf = confs[cat_id]

            __show_categories(ax, [cat_id], [cat_conf], colors, font_size)
        else:
            mask = confs > conf_threshold
            thresholded_categories = np.nonzero(mask)[0]
            thresholded_confs = confs[mask]

            # sort in desc order
            sort_indices = np.argsort(thresholded_confs)[::-1]
            thresholded_categories = thresholded_categories[sort_indices]
            thresholded_confs = thresholded_confs[sort_indices]

            __show_categories(ax, thresholded_categories, thresholded_confs, colors, font_size)

    plt.tight_layout()

def __show_categories(ax, categories, confs, colors, font_size, x=0, y=0):
    text = ''
    for cat, conf in zip(categories, confs):
        label_idx = int(cat - 1)# take background into account
        label = PascalVoc2012.CLASSES[label_idx] + ': ' + "{0:.2f}".format(conf)
        color = colors[label_idx]
        text = text + label + '\n'

    ax.text(x, y, text, verticalalignment='top', fontsize=font_size, bbox={'facecolor': color, 'alpha': 0.5})

def show_bboxes(images, bboxes, num_classes, cols, figsize=(12, 8)):
    if len(images) != len(bboxes):
        raise ValueError("The size of 'images' list should be equal to the size of 'bboxes' list")

    nb_images = len(images)
    rows = nb_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for img, boxes, (i, ax) in zip(images, bboxes, enumerate(axes.flat)):
        # when just one box is presented - convert it to array
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, 0)#np.array([boxes])

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

def show_predictions(images, predictions, num_classes, cols, conf_threshold=None, figsize=(12, 8), ax=None, font_size=14):
    if len(images) != len(predictions):
        raise ValueError("The size of 'images' list should be equal to the size of 'predictions' list")

    if conf_threshold is not None and not 0 <= conf_threshold <= 1:
        raise ValueError("'conf_threshold' should be either None or in range [0, 1]")

    nb_images = len(images)
    rows = nb_images // cols

    category_colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for img, img_preds, (i, ax) in zip(images, predictions, enumerate(axes.flat)):
        if img_preds.ndim == 1:
            img_preds = np.expand_dims(img_preds, 0)#np.array([img_preds])

        img_w, img_h = img.shape[1], img.shape[0]
        bbox_colors = plt.cm.hsv(np.linspace(0, 1, len(img_preds))).tolist()

        ax.imshow(img)

        for pred, bbox_color in zip(img_preds, bbox_colors):
            bbox = np.clip(pred[:4], 0, 1)
            confs = pred[4:]

            xmin, ymin, xmax, ymax = int(bbox[0] * img_w), int(bbox[1] * img_h), int(bbox[2] * img_w), int(bbox[3] * img_h)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=bbox_color, linewidth=2))

            if conf_threshold is None:
                cat_id = np.argmax(confs)
                cat_conf = confs[cat_id]

                __show_categories(ax, [cat_id], [cat_conf], category_colors, font_size, xmin, ymin)
            else:
                mask = confs > conf_threshold
                thresholded_categories = np.nonzero(mask)[0]
                thresholded_confs = confs[mask]

                # sort in desc order
                sort_indices = np.argsort(thresholded_confs)[::-1]
                thresholded_categories = thresholded_categories[sort_indices]
                thresholded_confs = thresholded_confs[sort_indices]

                __show_categories(ax, thresholded_categories, thresholded_confs, category_colors, font_size, xmin, ymin)

    plt.tight_layout()


def show_detections(img, predictions, num_classes, figsize=(12, 8), ax=None, font_size=14):
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
Prediction utilities
"""
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

"""
Freeze passed model starting from pointed layer
"""
def freeze_model(model, freeze_start_layer_name):
    #model.trainable = True
    trainable = False
    for layer in model.layers:
        if layer.name == freeze_start_layer_name:
            trainable = True
        layer.trainable = trainable


"""
Plot train/valid loss curves and save plot to the file ./loss_curve.png.
"""
def plot_loss(history, figsize=(12, 8), ax=None):
    plot_curve(history.history, ['loss', 'val_loss'], ['train', 'valid'], 'Model Loss', 'Loss', 'Epoch',
               figsize=figsize, ax=ax)

def plot_accuracy(history, figsize=(12, 8), ax=None):
    plot_curve(history.history, ['acc', 'val_acc'], ['train', 'valid'], 'Model Accuracy', 'Accuracy', 'Epoch',
               figsize=figsize, ax=ax)

def plot_loss_and_acc(history, figsize=(12, 8)):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    plot_loss(history, figsize=figsize, ax=axes[0])
    plot_accuracy(history, figsize=figsize, ax=axes[1])

def plot_losses(histories, model_names, figsize=(12, 8), ax=None):
    if len(histories) != len(model_names):
        raise ValueError("Length of 'histories' should be equal to length of 'model_names'")

    if not ax: fig, ax = plt.subplots(figsize=figsize)

    ax.set_title('Losses')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')

    for h, name in zip(histories, model_names):
        plot_curve(h.history, ['loss', 'val_loss'], ['train-'+name, 'valid-'+name], ax=ax)

def plot_accuracies(histories, model_names, figsize=(12, 8), ax=None):
    if len(histories) != len(model_names):
        raise ValueError("Length of 'histories' should be equal to length of 'model_names'")

    if not ax: fig, ax = plt.subplots(figsize=figsize)

    ax.set_title('Accuracies')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')

    for h, name in zip(histories, model_names):
        plot_curve(h.history, ['acc', 'val_acc'], ['train-'+name, 'valid-'+name], ax=ax)

def plot_losses_and_accuracies(histories, model_names, figsize=(12, 8)):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plot_losses(histories, model_names, axes[0])
    plot_accuracies(histories, model_names, axes[1])

def plot_curve(data, keys, data_titles, title=None, ylabel=None, xlabel=None, figsize=(12, 8), ax=None, out_file=None):
    if data_titles is None:
        data_titles = keys

    if len(data_titles) < len(keys):
        # use tail of keys as replacement for absent titles
        data_titles = data_titles + keys[len(data_titles):]

    if not ax: fig, ax = plt.subplots(figsize=figsize)
    if title: ax.set_title(title)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)

    for k, t in zip(keys, data_titles):
        ax.plot(data[k], label=t)

    ax.legend(loc='upper left')

    # save into file
    if out_file:
        plt.savefig(out_file)
    #plt.close()

    plt.tight_layout()