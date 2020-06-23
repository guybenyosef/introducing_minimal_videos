import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out):
        hinge_loss = 1 - out
        hinge_loss[hinge_loss < 0] = 0
        hinge_loss = 0.5 * (hinge_loss ** 2)
        return hinge_loss


class DrLIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.drlim = HingeLoss()

    def forward(self, emb1, emb2, gt):
        l2part = self.l2(emb1, emb2)
        return gt * l2part + (1 - gt) * self.drlim(l2part)


def load_loss(loss_name, weight=None):
    print('Using loss : %s..' % loss_name)
    if isinstance(loss_name, list):
        return [load_loss(l) for l in loss_name]

    if loss_name == 'L1':
        loss = nn.L1Loss()

    elif loss_name == 'SmoothL1':
        loss = nn.SmoothL1Loss()

    elif loss_name == 'L2':
        loss = nn.MSELoss()

    elif loss_name == 'BCE':  # binary_crossentropy
        loss = nn.BCELoss()

    elif loss_name == 'CrossEnt':  # categorical_crossentropy
        loss = nn.CrossEntropyLoss(weight=None)

    elif loss_name == 'WeightCrossEnt':  # categorical_crossentropy
        loss = nn.CrossEntropyLoss(weight=weight)

    elif loss_name == 'DRLIM':  # DRLIM http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        loss = DrLIM()


    else:
        print('ERROR: loss name does not exist..')
        return

    return loss
