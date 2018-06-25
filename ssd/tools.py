import math
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

from .data import PascalVoc2012
from .imaging import load_img, preprocess_img
from .layers import L2Normalize, PriorBox
from .losses import SsdLoss

CLASSES = np.array(PascalVoc2012.CLASSES)

def gtb_label(gtb):
    one_hot = gtb[4:]
    return CLASSES[one_hot == 1]

"""
Use this method to show original GTBs with categories.
"""
def show_gtbs(base_dir, test_files, global_gtbs, num_classes, cols=1, figsize=(12, 8), font_size=12):
    orig_images = []
    orig_gtbs = []
    for file in test_files:
        img = load_img(base_dir + file)
        orig_images.append(img)

        gtbs = global_gtbs[file].copy()
        # add background
        bg_col = np.zeros((len(gtbs), 1))
        gtbs = np.concatenate((gtbs[:, :4], bg_col[:], gtbs[:, 4:]), axis=1)

        orig_gtbs.append(gtbs)

    show_predictions(orig_images, orig_gtbs, num_classes, cols=cols, conf_threshold=None, figsize=figsize, font_size=font_size)

"""
Use this method to show the raw result of NN when only categories is predicted.
For example when LargestObjClassCodec is being used.
"""
def show_categories(images, predictions, num_classes, conf_threshold=None, cols=1, figsize=(12, 8), font_size=12):
    if len(images) != len(predictions):
        raise ValueError("The size of 'images' list should be equal to the size of 'predictions' list")

    if conf_threshold is not None and not 0 <= conf_threshold <= 1:
        raise ValueError("'conf_threshold' should be either None or in range [0, 1]")

    nb_images = len(images)
    rows = math.ceil(nb_images / cols)

    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for img, confs, (i, ax) in zip(images, predictions, enumerate(axes.flat)):
        ax.imshow(img)

        # get rid off background
        confs = confs[1:]

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
        label_idx = cat
        label = CLASSES[label_idx] + ': ' + "{0:.2f}".format(conf)
        color = colors[label_idx]
        prefix = '' if text == '' else '\n'
        text = prefix + text + label

    if text != '':
        ax.text(x, y, text, verticalalignment='top', fontsize=font_size, bbox={'facecolor': color, 'alpha': 0.5})

"""
Show normalized bboxes.
Use this method when LargestObjBoxCodec is being used.
bboxes are presented as corners - [xmin, ymin, xmax, ymax] normalized by image size
"""
def show_bboxes(images, bboxes, cols=1, figsize=(12, 8)):
    if len(images) != len(bboxes):
        raise ValueError("The size of 'images' list should be equal to the size of 'bboxes' list")

    nb_images = len(images)
    rows = math.ceil(nb_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for img, boxes, (i, ax) in zip(images, bboxes, enumerate(axes.flat)):
        # when just one box is presented - convert it to array
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, 0)#np.array([boxes])

        img_w, img_h = img.shape[1], img.shape[0]
        colors = plt.cm.hsv(np.linspace(0, 1, len(boxes) + 1)).tolist()

        ax.imshow(img)

        for bbox, color in zip(boxes, colors):
            xmin = int(bbox[0] * img_w)
            ymin = int(bbox[1] * img_h)
            xmax = int(bbox[2] * img_w)
            ymax = int(bbox[3] * img_h)

            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            patch = ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))

"""
Use this method to show the raw output of NN.
Appropriate when BBoxCodec or LargestObjBoxAndClassCodec are being used.
"""
def show_predictions(images, predictions, num_classes, conf_threshold=None, cols=1, figsize=(12, 8), font_size=12):
    if len(images) != len(predictions):
        raise ValueError("The size of 'images' list should be equal to the size of 'predictions' list")

    if conf_threshold is not None and not 0 <= conf_threshold <= 1:
        raise ValueError("'conf_threshold' should be either None or in range [0, 1]")

    nb_images = len(images)
    rows = math.ceil(nb_images / cols)

    category_colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    figsize=(figsize[0], figsize[1]*rows)
    #print('figsize: {}'.format(figsize))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if nb_images == 1:
        axes = np.array([axes])

    for img, img_preds, (i, ax) in zip(images, predictions, enumerate(axes.flat)):
        if img_preds.ndim == 1:
            img_preds = np.expand_dims(img_preds, 0)#np.array([img_preds])

        img_w, img_h = img.shape[1], img.shape[0]
        bbox_colors = plt.cm.hsv(np.linspace(0, 1, len(img_preds) + 1)).tolist()

        ax.imshow(img)

        for pred, bbox_color in zip(img_preds, bbox_colors):
            bbox = np.clip(pred[:4], 0, 1)
            confs = pred[5:4+num_classes]# skip background

            xmin, ymin, xmax, ymax = int(bbox[0] * img_w), int(bbox[1] * img_h), int(bbox[2] * img_w), int(bbox[3] * img_h)

            add_rect = False

            if conf_threshold is None:
                add_rect = True
                cat_id = np.argmax(confs)
                cat_conf = confs[cat_id]

                __show_categories(ax, [cat_id], [cat_conf], category_colors, font_size, xmin, ymin)
            else:
                mask = confs > conf_threshold
                if np.sum(mask) > 0:
                    add_rect = True
                    thresholded_categories = np.nonzero(mask)[0]
                    thresholded_confs = confs[mask]

                    # sort in desc order
                    sort_indices = np.argsort(thresholded_confs)[::-1]
                    thresholded_categories = thresholded_categories[sort_indices]
                    thresholded_confs = thresholded_confs[sort_indices]

                    __show_categories(ax, thresholded_categories, thresholded_confs, category_colors, font_size, xmin, ymin)

            if add_rect:
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=bbox_color, linewidth=2))

    plt.tight_layout()

"""
Use this method to show the result built by Detection.detect_bboxes
How to use:
detector = Detection(num_classes, target_img_size, bbox_codec, out_nms_top_k=5, out_conf_threshold=0.5)
imgs, predictions = detector.detect_bboxes(model, test_images)
show_detections(test_images, predictions, cols=3, figsize=(12, 4))
"""
def show_detections(images, predictions, cols, figsize=(12, 8), font_size=12):
    nb_images = len(images)
    rows = math.ceil(nb_images / cols)

    figsize = (figsize[0], figsize[1] * rows)
    # print('figsize: {}'.format(figsize))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if nb_images == 1:
        axes = np.array([axes])

    for img, img_preds, (i, ax) in zip(images, predictions, enumerate(axes.flat)):
        if img_preds.ndim == 1:
            img_preds = np.expand_dims(img_preds, 0)  # np.array([img_preds])

        img_w, img_h = img.shape[1], img.shape[0]
        bbox_colors = plt.cm.hsv(np.linspace(0, 1, len(img_preds) + 1)).tolist()

        ax.imshow(img)
        #ax.set_xticks(np.linspace(0, img_w, 8))
        #ax.set_yticks(np.linspace(0, img_h, 8))
        #ax.grid()

        for pred, bbox_color in zip(img_preds, bbox_colors):
            # [class, conf, xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = int(pred[2]), int(pred[3]), int(pred[4]), int(pred[5])
            label_idx = int(pred[0] - 1)
            label = CLASSES[label_idx] + ': ' + "{0:.2f}".format(pred[1])
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1

            patch = ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=bbox_color, linewidth=2))
            text = ax.text(xmin, ymin, label, verticalalignment='top',
                           fontsize=font_size, bbox={'facecolor': bbox_color, 'alpha': 0.5})


"""
How to use:
bbox_codec = BBoxCodec(prior_boxes, num_classes, iou_threshold=0.5, encode_variances=True, match_per_prediction=True)
gen = Generator(gtb, img_dir, target_img_size, DataAugmenter(), bbox_codec)
test_generator = gen.flow(train_samples, batch_size=8, do_augment=False, return_debug_info=True)
x, y, orig_x, orig_y, matches = next(test_generator)
show_matches(orig_x, orig_y, matches, prior_boxes, cols=2, figsize=(12,6))
"""
def show_matches(images, gtbs, matches, prior_boxes, cols=1, figsize=(12, 8), font_size=12):
    nb_images = len(images)
    rows = math.ceil(nb_images / cols)

    figsize = (figsize[0], figsize[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if nb_images == 1:
        axes = np.array([axes])

    for img, image_gtbs, image_matches, (i, ax) in zip(images, gtbs, matches, enumerate(axes.flat)):
        img_w, img_h = img.shape[1], img.shape[0]
        ax.imshow(img)

        # color per gtb
        colors = plt.cm.hsv(np.linspace(0, 1, len(image_gtbs) + 1)).tolist()

        for gtb_idx, gtb in enumerate(image_gtbs):
            color = colors[gtb_idx]

            xmin, ymin, xmax, ymax = int(gtb[0] * img_w), int(gtb[1] * img_h), int(gtb[2] * img_w), int(gtb[3] * img_h)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
            ax.text(xmin, ymin, gtb_label(gtb), verticalalignment='top', fontsize=font_size, bbox={'facecolor': color, 'alpha': 0.5})

            matched_pbs = prior_boxes[image_matches == gtb_idx]
            for pb in matched_pbs:
                xmin, ymin, xmax, ymax = int(pb[0] * img_w), int(pb[1] * img_h), int(pb[2] * img_w), int(pb[3] * img_h)
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))

    plt.tight_layout()


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
        img = load_img(base_dir + file)
        orig_images.append(img)
        preprocessed_images.append(preprocess_img(img, target_img_size))

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

def load_ssd_model(filepath, num_classes, hard_neg_pos_ratio):
    return load_model(filepath, custom_objects={'L2Normalize': L2Normalize, 'PriorBox': PriorBox,
                                                'loss': SsdLoss(num_classes=num_classes, hard_neg_pos_ratio=hard_neg_pos_ratio).loss})

"""
Plot train/valid loss curves and save plot to the file ./loss_curve.png.
"""
def plot_loss(history, figsize=(12, 8), ax=None, out_file=None):
    plot_curve(history.history, ['loss', 'val_loss'], ['train', 'valid'], 'Model Loss', 'Loss', 'Epoch',
               figsize=figsize, ax=ax, out_file=out_file)

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