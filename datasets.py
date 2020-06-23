import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import CONSTS
import os
import numpy as np
from PIL import Image
import time

from utils import make_dataset

class ActionOrNot(data.Dataset):
    def __init__(self, type0_pathname=None, type1_pathname=None, input_transform=None):

        # load file names:
        if type0_pathname is not None:
            self.type0_examples = make_dataset(type0_pathname)
        else:
            self.type0_examples = []
        #
        if type1_pathname is not None:
            self.type1_examples = make_dataset(type1_pathname)
        else:
            self.type1_examples = []
        #
        self.image_filenames = self.type0_examples + self.type1_examples
        self.labels = [0] * len(self.type0_examples) + [1] * len(self.type1_examples)

        # pos/neg inds:
        self.inds_type0_examples = list(np.where(np.array(self.labels) == 0)[0])
        self.inds_type1_examples = list(np.where(np.array(self.labels) == 1)[0])

        # pos/neg ratio:
        if type0_pathname is not None and type1_pathname is not None:
            self.class_ratio = [len(self.type0_examples)/len(self.image_filenames),
                                 len(self.type1_examples)/len(self.image_filenames)]
        else:
            self.class_ratio = None

        # num frames:
        self.num_of_frames = 8  # 16

        # basic transform:
        self.input_transform = transforms.Compose(
            [
                transforms.Resize(size=(116, 116)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, img_idx):

        filename = self.image_filenames[img_idx]
        vid = Image.open(filename)
        frames = []
        for findx in range(self.num_of_frames):
            vid.seek(findx % 2)
            image = vid.copy().convert('RGB')
            image = self.input_transform(image)
            frames.append(image)

        one_hotlabel = np.zeros(2)
        one_hotlabel[self.labels[img_idx]] = 1.0
        label = self.labels[img_idx]

        return (torch.stack(frames, dim=1),
                torch.tensor(label, dtype=torch.long),
                filename)

class datas(object):
    def __init__(self, trainset=None, testset=None, num_classes=None):
        self.trainset = trainset
        self.testset = testset
        self.num_classes = num_classes


# ===================
# load:
# ===================

def load_dataset(ds_name, negs_set=1):

    data_dir_name = CONSTS.DATA_DIR
    negatives_dir_name = CONSTS.NEGATIVES_DIR
    negs_set = str(negs_set)

    if ds_name == 'RowingOrNot':
        num_classes = 2
        train_type0_path = os.path.join(negatives_dir_name, 'nonrowing', negs_set)
        train_type1_path = os.path.join(data_dir_name, 'ROWING/train')
        test_type0_path = os.path.join(negatives_dir_name, 'nonrowing', '0')
        test_type1_path = os.path.join(data_dir_name, 'ROWING/test')
        trainset = ActionOrNot(train_type0_path, train_type1_path, input_transform=None)
        testset = ActionOrNot(test_type0_path, test_type1_path, input_transform=None)

    else:
        print('ERROR: dataset name does not exist..')
        return

    print('loading dataset : %s (%d classes).. number of train examples is %d, number of test examples is %d.'
          % (ds_name, num_classes, len(trainset), len(testset)))

    return datas(trainset, testset, num_classes)


if __name__ == '__main__':

    ds = load_dataset('RowingOrNot')
    #ds = load_dataset('FourMIRCs')
    img, label, filename = ds.testset.__getitem__(250)

    im = TF.to_pil_image(img)
    im.show()
    print(filename)
    print(label)
    time.sleep(10)
