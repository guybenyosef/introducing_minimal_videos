import os
import sys
import torch
import numpy as np
from train import identify_model, run_epoch, eval_model
from datasets import ActionOrNot
from models import load_model
from losses import load_loss
from sacred import Experiment

ex = Experiment('MinimalVideos')

@ex.config
def config():
    batch_size = None
    weights = 'storage/logs/RowingOrNot/3/ResNet3D18/5/weights_RowingOrNot_ResNet3D18_best.pth'
    subset = None
    type0 = '/shared-data5/guy/data/minimal/negatives_video/nonrowing/2'
    type1 = 'dnn_data/ROWING/test/'
    num_workers = 32
    plot_flag = False
    num_examples_to_plot = 0
    testid = 'naive'

@ex.named_config
def hard():
    type0 = 'hardneg/hardneg_rowing'
    testid = 'hard'

@ex.named_config
def veryhard():
    type0 = 'hardneg/veryhardneg_rowing'
    testid = 'veryhard'

@ex.named_config
def submirc():
    type0 = 'dnn_data/ROWING/spatial_submirc_rowing'
    testid = 'submirc'

@ex.named_config
def plot():
    num_examples_to_plot = 100

@ex.named_config
def plotall():
    num_examples_to_plot = 100000

#############################
# Load the Model
#############################
def load_the_model(weights):

    # load stored weights:
    if (weights is not None) and (os.path.exists(weights)):
         print("Loading weights file %s.." % weights)
         checkpoint = torch.load(weights, map_location='cpu')
         batch_size_at_train = checkpoint['args']['batch_size']
         model = load_model(checkpoint['args']['model'])
         criterion = load_loss(checkpoint['args']['loss'])
         print('Testing model {}'.format(model.__class__.__name__))
         model.load_state_dict(checkpoint['model_state_dict'])
         identify_model(checkpoint=checkpoint)
    else:
        sys.exit("Error: weights file does not exist")

    # set eval mode:
    model.eval()

    # set dirname, basename:
    dirname = os.path.dirname(weights)
    basename = os.path.splitext(os.path.basename(weights))[0]

    return model, basename, dirname, criterion, batch_size_at_train

#############################
# Run Test Procedure:
#############################
def run_test(model, basename, batch_size, criterion, subset, outname, device, inputdir_type0, inputdir_type1, num_examples_to_plot=100, num_workers=32):

    # For either multi or binary classifier use:
    testset = ActionOrNot(type0_pathname=inputdir_type0, type1_pathname=inputdir_type1)

    loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, pin_memory=True,
                                         num_workers=num_workers)
    print('Testing %d image examples..' % len(testset))

    if subset is not None:
        type1 = np.asarray(testset.inds_type1_examples)  # all pos
        type0 = np.random.choice(testset.inds_type0_examples, subset)  # subset neg
        loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(np.concatenate((type1, type0))),
                                                  shuffle=False, pin_memory=True,
                                                  num_workers=num_workers)

    test_loss, test_acc, test_auc, misc = run_epoch(epoch=1, loader=loader, optimizer=None, model=model, criterion=criterion, device=device, prefix='Testing')
    test_total_outputs_class, test_total_gt_labels, test_total_filenames = misc[0], misc[1], misc[2]

    filename = os.path.join(outname, 'misclassifications_{}'.format(basename))
    eval_model(test_total_outputs_class, test_total_gt_labels, test_total_filenames,
               outname=filename, num_examples_to_plot=num_examples_to_plot)



########################################
########################################
# Main
########################################
########################################
@ex.automain
def main(_run):
    # INTRO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD model:
    model, basename, dirname, criterion, batch_size_at_train = load_the_model(_run.config['weights'])
    model.to(device)

    if _run.config['batch_size'] is None:
        batch_size = batch_size_at_train
    else:
        batch_size = _run.config['batch_size']

    outname = os.path.join(dirname, 'eval')
    if not os.path.exists(outname):
        os.makedirs(outname)

    basename = basename + "_testid_" + _run.config['testid']

    run_test(model, basename, batch_size, criterion, _run.config['subset'], outname, device,
             inputdir_type0=_run.config['type0'],  inputdir_type1=_run.config['type1'],
             num_examples_to_plot=_run.config['num_examples_to_plot'], num_workers=_run.config['num_workers'])



