import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.metrics import confusion_matrix

import re
import itertools
import matplotlib
import numpy as np
from textwrap import wrap


def imshow(img):
    img = img / 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def log_conf_matrix(G, correct_labels, predict_labels, labels, step, title='Confusion matrix', tensor_name ='MyFigure/image', normalize=False):
    cm = confusion_matrix(correct_labels, predict_labels) #, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)

    fig = matplotlib.figure.Figure(figsize=(5, 5), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))
    fontsize = 24 - round(len(classes) / (1.5 if len(classes) > 13 else 0.8))

    ax.set_xlabel('Predicted', fontsize=fontsize)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=fontsize,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=fontsize)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=fontsize, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=fontsize, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    agg = fig.canvas.switch_backends(FigureCanvasAgg)
    agg.draw()

    img = np.fromstring(agg.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(agg.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    G.writer.add_image('confusion_matrix', img, step)
    plt.close(fig)
