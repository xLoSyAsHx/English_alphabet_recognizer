import torch

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from scipy import interp

import re
import sys
import itertools
import matplotlib
import numpy as np
from textwrap import wrap


class MLMetrics:
    def __init__(self, cm, pred_list, lbl_list, prob_list):
        self.cm = cm
        self.pred_list = pred_list
        self.lbl_list  = lbl_list
        self.prob_list = prob_list

        self.FP = FP = cm.sum(axis=0) - np.diag(cm)
        self.FN = FN = cm.sum(axis=1) - np.diag(cm)
        self.TP = TP = np.diag(cm)
        self.TN = TN = cm.sum() - (FP + FN + TP)

        eps = sys.float_info.min
        self.TPR = TP / (TP + FN + eps) # recall (true positive rate)
        self.TNR = TN / (TN + FP + eps)
        self.PPV = TP / (TP + FP + eps) # precision (positive prediction value)
        self.NPV = TN / (TN + FN + eps)
        self.FPR = FP / (FP + TN + eps)
        self.FNR = FN / (TP + FN + eps)
        self.FDR = FP / (TP + FP + eps)
        self.F1  = 2 * (self.PPV * self.TPR) / (self.PPV + self.TPR + eps)
        self.ACC = (TP + TN) / (TP + FP + FN + TN + eps)


def imshow(img):
    img = img / 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:2.0f}' if height < 1.0 else f'{height:3.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def add_fig_to_tensorboard(G, fig, tag, step, close=True):
    agg = fig.canvas.switch_backends(FigureCanvasAgg)
    agg.draw()

    img = np.fromstring(agg.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(agg.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    G.writer.add_image(tag, img, step)
    if close:
        plt.close(fig)


def calculate_metrics(G, net, test_loader, epoch, log_loss=True):
    with torch.no_grad():
        pred_list = torch.zeros(0, dtype=torch.long,  device='cpu')
        lbl_list  = torch.zeros(0, dtype=torch.long,  device='cpu')
        prob_list = torch.zeros(0, dtype=torch.float, device='cpu')
        for i, data in enumerate(test_loader):
            images, labels = data[0].to(G.device), data[1].to(G.device)

            outputs = net(images)
            if log_loss:
                G.writer.add_scalar('Loss/test', G.criterion(outputs, labels).item(), epoch)

            p_values, p_indexes = torch.max(outputs.data, 1)
            pred_list = torch.cat([pred_list, p_indexes.view(-1).cpu()])
            lbl_list  = torch.cat([lbl_list, labels.view(-1).cpu()])
            prob_list = torch.cat([prob_list, p_values.view(-1).cpu()])

        cm = confusion_matrix(lbl_list, pred_list)
        return MLMetrics(cm, pred_list, lbl_list, prob_list)


# G - global variables
# M - metrics
def log_conf_matrix(G, M, classes, step, title='Confusion matrix', tensor_name ='MyFigure/image', normalize=False):
    cm = M.cm
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200, facecolor='w', edgecolor='k')
    ax.imshow(M.cm, cmap='Oranges')

    tick_marks = np.arange(len(classes))
    fontsize = 24 - round(len(classes) / (1.5 if len(classes) > 13 else 0.8))

    ax.set_xlabel('Predicted', fontsize=fontsize)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=fontsize, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=fontsize)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=fontsize, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    ax.set_title('Confusion matrix')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=fontsize, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    add_fig_to_tensorboard(G, fig, 'confusion_matrix', step)


# G - global variables
# M - metrics
def log_TPR_PPV_F1_bars(G, M, classes, step):
    fontsize = 24 - round(len(classes) / (1.5 if len(classes) > 13 else 0.8))

    total_width = 0.75 * 2
    el_width = total_width / 3
    x = np.arange(0, len(classes) * 2, 2)
    np.set_printoptions(precision=2)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200, facecolor='w', edgecolor='k')
    rects_TPR = ax.bar(x - total_width / 2 + el_width * 0.5, M.TPR * 100, width=el_width, label='Recall')
    rects_PPV = ax.bar(x - total_width / 2 + el_width * 1.5, M.PPV * 100, width=el_width, label='Precision')
    rects_F1  = ax.bar(x - total_width / 2 + el_width * 2.5, M.F1  * 100, width=el_width, label='F1 score', color='r')

    ax.set_ylim([0, 110])
    ax.set_xlabel('Classes', fontsize=fontsize + 2)
    ax.set_ylabel('Percents', fontsize=fontsize + 2)
    ax.set_xticklabels(classes, fontsize=fontsize, ha='center')
    ax.set_title('Recall / Precision / F1')
    ax.grid()
    ax.legend(loc='best')

    autolabel(rects_TPR, ax)
    autolabel(rects_PPV, ax)
    autolabel(rects_F1,  ax)

    fig.set_tight_layout(True)
    add_fig_to_tensorboard(G, fig, 'recall/precision/F1', step)


# G - global variables
# M - metrics
def log_ROC_AUC(G, M, classes, step):
    fpr = dict()
    tpr = dict()
    interp_tprs = []
    roc_auc = dict()
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(
            M.pred_list[M.lbl_list == i] == M.lbl_list[M.lbl_list == i],
            M.prob_list[M.lbl_list == i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        interp_tpr = interp(mean_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    np.set_printoptions(precision=2)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200, facecolor='w', edgecolor='k')

    for i in range(len(classes)):
        ax.plot(fpr[i], tpr[i], label=f"ROC '{classes[i]}' (AUC = {roc_auc[i]:0.2f})", alpha=0.5)
    ax.plot([0, 1], [0, 1],     label='Chance', color='r', linestyle='--')

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([roc_auc[key] for key in roc_auc])
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=rf'Mean ROC (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})', lw=2, alpha=0.8)

    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curves')
    ax.grid()
    ax.legend(loc='lower right')

    fig.set_tight_layout(True)
    add_fig_to_tensorboard(G, fig, 'ROC-AUC', step)
