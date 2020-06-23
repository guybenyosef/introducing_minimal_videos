import torch
import torchvision
import numpy as np
from pathlib import Path
import cv2
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from sklearn import metrics
from torch.utils.tensorboard.summary import hparams
import shutil
from shutil import copy
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import scipy.misc
import json
import scipy.io

#from Digital_Inspections.utils import compute_gauge_value
#from utils_kpts import get_kpts, draw_paint

def make_dataset(dir, ext='gif'):
    nparrays = []
    for fname in Path(dir).glob('**/*.' + ext):
        nparrays.append(str(fname))

    return nparrays


def make_dataset_txtfile(filename):
    f = open(filename, "r")
    #nparrays = f.readlines()
    nparrays = f.read().splitlines()
    f.close()

    return nparrays


def better_hparams(writer, hparam_dict=None, metric_dict=None):
    """Add a set of hyperparameters to be compared in TensorBoard.
    Args:
        hparam_dict (dictionary): Each key-value pair in the dictionary is the
          name of the hyper parameter and it's corresponding value.
        metric_dict (dictionary): Each key-value pair in the dictionary is the
          name of the metric and it's corresponding value. Note that the key used
          here should be unique in the tensorboard record. Otherwise the value
          you added by `add_scalar` will be displayed in hparam plugin. In most
          cases, this is unwanted.

        p.s. The value in the dictionary can be `int`, `float`, `bool`, `str`, or
        0-dim tensor
    Examples::
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter() as w:
            for i in range(5):
                w.add_hparams({'lr': 0.1*i, 'bsize': i},
                              {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
    Expected result:
    .. image:: _static/img/tensorboard/add_hparam.png
       :scale: 50 %
    """
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError('hparam_dict and metric_dict should be dictionary.')
    exp, ssi, sei = hparams(hparam_dict, metric_dict)

    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    # writer.file_writer.add_summary(sei)
    # for k, v in metric_dict.items():
    #     writer.add_scalar(k, v)
    # with SummaryWriter(log_dir=os.path.join(self.file_writer.get_logdir(), str(time.time()))) as w_hp:
    #     w_hp.file_writer.add_summary(exp)
    #     w_hp.file_writer.add_summary(ssi)
    #     w_hp.file_writer.add_summary(sei)
    #     for k, v in metric_dict.items():
    #         w_hp.add_scalar(k, v)

    return sei


#############################
# Compute AUC (or AP)
#############################
def compute_auc_multiclass(predicted_labels, groundtruth_labels, max_num_labels=5):
    '''
    compute AUC
    :param W: <int> predicted_labels, <int> groundtruth_labels
    :return: <numpy> mean Average Precision (a.k.a., Area Under Curve)
    '''

    auc = np.full(max_num_labels, np.nan)

    for class_indx in range(1,5):
        if (0 in groundtruth_labels) and (class_indx in groundtruth_labels):
            noclass_inds = np.where(np.array(predicted_labels) == 0)[0]
            yesclass_inds = np.where(np.array(predicted_labels) == class_indx)[0]
            relevant_indices = np.concatenate((noclass_inds, yesclass_inds), axis=0)
            relevant_gt_labels = [0] * len(relevant_indices)
            relevant_outputs_class = [0] * len(relevant_indices)
            for ii in range(len(relevant_indices)):
                if groundtruth_labels[relevant_indices[ii]] > 0:
                    relevant_gt_labels[ii] = 1
                if predicted_labels[relevant_indices[ii]] > 0:
                    relevant_outputs_class[ii] = 1

            auc[class_indx] = metrics.roc_auc_score(y_true=relevant_gt_labels, y_score=relevant_outputs_class)
            print("AUC [0 vs. %d] is %.6f" % (class_indx, auc[class_indx]))

    mAP = auc[~np.isnan(auc)].mean()

    print("Mean Average Percision (mAP) is %.6f " % mAP)

    return mAP

####################
# utils Handling IMAGES
####################
TorchToPIL = torchvision.transforms.ToPILImage()
PILtoTorch = torchvision.transforms.ToTensor()

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

# def im_to_numpy(img):
#     img = to_numpy(img)
#     img = np.transpose(img, (1, 2, 0)) # H*W*C
#     return img
#
# def im_to_torch(img):
#     img = np.transpose(img, (2, 0, 1)) # C*H*W
#     img = to_torch(img).float()
#     if img.max() > 1:
#         img /= 255
#     return img

# from: https://github.com/namedBen/Convolutional-Pose-Machines-Pytorch/blob/master/train_val/Mytransforms.py
def normalize(tensor, mean, std):
    """Normalize a ``torch.tensor``
    Args:
        tensor (torch.tensor): tensor to be normalized.
        mean: (list): the mean of BGR
        std: (list): the std of BGR

    Returns:
        Tensor: Normalized tensor.
    """
    # (Mytransforms.to_tensor(img), [128.0, 128.0, 128.0], [256.0, 256.0, 256.0]) mean, std

    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def denormalize(tensor, mean, std):
    """Normalize a ``torch.tensor``
    Args:
        tensor (torch.tensor): tensor to be normalized.
        mean: (list): the mean of BGR
        std: (list): the std of BGR

    Returns:
        Tensor: deNormalized tensor.
    """
    # (Mytransforms.to_tensor(img), [128.0, 128.0, 128.0], [256.0, 256.0, 256.0]) mean, std

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

####################
# utils PLOTS
####################

def plot_test_img2type(fig, filenames, pred_label, gt_label):
    # plot:
    num_examples_to_plot = 16

    for indx in range(min(len(filenames), num_examples_to_plot)):
        ax = fig.add_subplot(4, 4, indx + 1)
        img = cv2.imread(filenames[indx])
        img = cv2.resize(img, (100, 100))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title('pr:%d, gt:%d' % (pred_label[indx], gt_label[indx]), fontsize=40)
        #suptitle('test title', fontsize=20)

    return fig


def plot_misclassifications_grid(files, labels):

    new_im = Image.new('RGB', (410, 410))

    index = 0
    idddxxx = np.random.randint(0, len(files), min(100, len(files)))
    for i in range(10, 400, 40):
        for j in range(10, 400, 40):
            if index < len(files):
                im = Image.open(files[idddxxx[index]])
                lb = labels[idddxxx[index]]
                im.thumbnail((30, 30))
                draw = ImageDraw.Draw(im)
                #draw.text((0, 0), str(lb), fill=128)
                draw.text((0, 0), str(lb), fill=0) #color=(255, 0, 255))#'magenta')
                new_im.paste(im, (i, j))
                index += 1


    #new_im.show()
    #input("Press Enter to continue...")

    return new_im


def copy_misclassifications(files, labels, outfolder, limit=100):

    # setting output folders and files:
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)
    os.makedirs(outfolder)

    # index = 0
    # idddxxx = range(len(files))#np.random.randint(0, len(files), min(limit, len(files)))
    #
    # if index < len(files):
    #     copy(files[idddxxx[index]], outfolder) #os.path.join(outfolder, str(labels[idddxxx[index]])))
    #     index += 1
    for file in files[:limit]:
        copy(file, outfolder)

# def copy_files_from_textfile(textfile, outfolder, limit=None):
#     num_neg_files = 0
#     # setting output folders and files:
#     if os.path.exists(outfolder):
#         shutil.rmtree(outfolder)
#     os.makedirs(outfolder)
#
#     for line in open(textfile):
#         ll = line.split(',')
#         filename = ll[0]
#         gt_label = int(ll[1])
#         if gt_label == 0:
#             num_neg_files += 1
#             copy(filename, outfolder)
#     print("%d neg files were written to %s" % (num_neg_files, outfolder))
