"""
MIT License
"""
### Import external libraries
import torch
import torch.optim as optim
import os
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import socket
from sacred import Experiment
from sacred.observers import FileStorageObserver

### Import internal libraries
from models import load_model
from losses import load_loss
from datasets import load_dataset
from utils import compute_auc_multiclass, better_hparams, plot_misclassifications_grid, to_numpy, copy_misclassifications

ex = Experiment('MinimalVideos')
logs_dir = 'storage'
ex.observers.append(FileStorageObserver(logs_dir))
def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if 'it/s]' not in line)
    return '\n'.join(lines)
ex.captured_out_filter = remove_progress

@ex.config
def config():
    model = 'ResNet3D18' # DNN model to train
    loss = 'CrossEnt' # Loss function for training the model
    dataset = 'RowingOrNot' # Dataset to load for training
    inputsize = 112
    batch_size = 32 #100
    num_workers = 32
    epochs = 10  # 'Specify the number of epochs to train'
    optimizer = 'ADAM' # Specify the optimizer
    learningrate = 1e-5
    seed = 1234
    subset = None
    negs_set = 1
    overfit = False
    evaluation_interval = 1

@ex.named_config
def overfit():
    overfit = True
    negs_set = 0
    epochs = 300

@ex.named_config
def equal():
    subset = 100
    epochs = 1000
    evaluation_interval = 10
    loss = 'CrossEnt'

@ex.named_config
def mini():
    subset = 1000
    epochs = 100
    evaluation_interval = 1
    loss = 'WeightCrossEnt'

@ex.named_config
def weighted():
    loss = 'WeightCrossEnt'

# ===========================
# ===========================
# Utils
# ===========================
# ===========================
def save_model(filename, epoch, model, args, current_train_loss, current_val_loss, best_val_metric, current_val_accuracy, hostname):

    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'epoch': epoch,
        'best_val_metric': best_val_metric,
        'current_train_loss': current_train_loss,
        'current_val_loss': current_val_loss,
        'current_val_accuracy': current_val_accuracy,
        'hostname': hostname,
        'saved_to': filename,

    }, filename)

    print("model saved to {}".format(filename))


def identify_model(filename=None, checkpoint=None):

    if filename is not None and os.path.exists(filename):
            print("loading file %s.."% filename)
            checkpoint = torch.load(filename)

    if checkpoint is not None:

        print('epoch is %d' % checkpoint['epoch'])
        if 'args' in checkpoint:
            print(checkpoint['args'])
        if 'current_train_loss' in checkpoint:
            print('current_train_loss = %.6f' % checkpoint['current_train_loss'])
        if 'current_val_loss' in checkpoint:
            print('current_val_loss = %.6f' % checkpoint['current_val_loss'])
        if 'best_val_loss' in checkpoint:
            print('best_val_metric = %.6f' % checkpoint['best_val_metric'])
        if 'current_val_accuracy' in checkpoint:
            print('current_val_accuracy = %.6f' % checkpoint['current_val_accuracy'])
        if 'hostname' in checkpoint:
            print('Host name: %s' % checkpoint['hostname'])
        if 'saved_to' in checkpoint:
            print('saved_to = %s' % checkpoint['saved_to'])

    else:
        print("path names or checkpoints do not exist..")

########################################
########################################
# Data
########################################
########################################
def sample_dataset(trainset, valset, overfit, subset, batch_size, num_workers, verbose=False):
    if overfit:  # sample identical very few examples for both train ans val sets:
        num_samples_for_overfit = 10
        type1 = np.random.choice(trainset.inds_type1_examples, num_samples_for_overfit)
        type0 = np.random.choice(trainset.inds_type0_examples, num_samples_for_overfit)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(np.concatenate((type1, type0))),
                                                  shuffle=False, pin_memory=True)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(np.concatenate((type1, type0))),
                                                 shuffle=False, pin_memory=True)
        nott = ("DATA: Sampling identical sets of %d POS and %d NEG examples for train and val sets.. " % (num_samples_for_overfit, num_samples_for_overfit))

    elif subset is not None:
        # Train:
        type1 = np.asarray(trainset.inds_type1_examples)  # all pos
        type0 = np.random.choice(trainset.inds_type0_examples, subset)  # subset neg
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(np.concatenate((type1, type0))),
                                                  shuffle=False, pin_memory=True,
                                                  num_workers=num_workers)
        # Val:
        type1 = np.asarray(valset.inds_type1_examples)  # all pos
        type0 = np.asarray(valset.inds_type0_examples)  # all neg
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(np.concatenate((type1, type0))),
                                                 shuffle=False, pin_memory=True,
                                                 num_workers=num_workers)
        nott = ("DATA: Sampling all POS and %d NEG examples for train and val sets.. " % subset)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, pin_memory=True,
                                                  num_workers=num_workers)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                 shuffle=True, pin_memory=True,
                                                 num_workers=num_workers)
        nott = ("DATA: Sampling all POS and all NEG examples for train and val sets.. ")

    if verbose:
        print(nott)

    return trainloader, valloader

########################################
########################################
# Run epoch for Train/Validate/Eval
########################################
########################################
def run_epoch(epoch, loader, optimizer, model, criterion, device, prefix):

    # Init:
    total_loss, total_accuracy = 0.0, 0.0
    total_predicted_class, total_gt_labels, total_filenames = [], [], []

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_idx, batch_data in enumerate(loader):

            # zero the parameter gradients
            if optimizer is not None:
                model.zero_grad()

            # Extract batch data:
            if device.type == 'cpu':
                images, labels = batch_data[:2]
            else:
                images, labels = [d.cuda() for d in batch_data[:2]]
            filenames = batch_data[2]

            # forward
            predicted_class_likelihood = model(images)
            loss = criterion(predicted_class_likelihood, labels)
            # backward
            if optimizer is not None: # in train mode
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            _, predicted_class = torch.max(predicted_class_likelihood, 1)
            total_accuracy += (predicted_class == labels).float().sum().item()
            #
            predicted_class = to_numpy(predicted_class)
            total_predicted_class = total_predicted_class + list(predicted_class)
            labels = to_numpy(labels)
            total_gt_labels = total_gt_labels + list(labels)
            total_filenames = total_filenames + list(filenames)

            del loss
            pbar.update()

    auc = compute_auc_multiclass(predicted_labels=total_predicted_class, groundtruth_labels=total_gt_labels, max_num_labels=2)
    misc = [total_predicted_class, total_gt_labels, total_filenames]

    return total_loss / len(total_filenames), total_accuracy / len(total_filenames), auc, misc

########################################
########################################
# Eval procedure
########################################
########################################
def eval_model(total_outputs_class, total_gt_labels, total_filenames, outname, num_examples_to_plot=100):
    # analyze results
    is_correct = np.array(total_outputs_class, dtype=int) == np.array(total_gt_labels, dtype=int)
    total_test_accuracy = sum(is_correct) / len(total_filenames)
    incorrect_indices = np.where(is_correct == 0)[0]
    incorrect_files = [total_filenames[i] for i in incorrect_indices]
    incorrect_files_pred_classes = [total_outputs_class[i] for i in incorrect_indices]
    incorrect_files_gt_classes = [total_gt_labels[i] for i in incorrect_indices]
    print("Number of incorrect detections is %d out of %d. Classification accuracy is.. %.6f"
          % (len(incorrect_files), len(total_filenames), total_test_accuracy))

    if outname is not None:
        # plot
        out_grid = plot_misclassifications_grid(incorrect_files, incorrect_files_pred_classes)
        out_grid.save(outname + ".png")
        # copy
        copy_misclassifications(incorrect_files, incorrect_files_pred_classes, outname, limit=num_examples_to_plot)
        print("misclassified images were saved as %s" % outname)
        # store names
        textfilename = outname + ".txt"
        with open(textfilename, 'w') as f:
            for indx, item in enumerate(incorrect_files):
                f.write("%s,%d,%d\n" % (item, incorrect_files_gt_classes[indx], incorrect_files_pred_classes[indx]))
        print("misclassified files were written to %s" % textfilename)

# ===========================
# ===========================
# Main
# ===========================
# ===========================
@ex.automain
def main(_run):
    print(_run.config)
    torch.manual_seed(_run.config['seed'])
    np.random.seed(_run.config['seed'])

    hostname = socket.getfqdn()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Training on machine name %s..' % hostname)
    print(device)

    # load data:
    ds = load_dataset(_run.config['dataset'], negs_set=_run.config['negs_set'])
    trainloader, valloader = sample_dataset(ds.trainset, ds.testset, _run.config['overfit'], _run.config['subset'],
                                            _run.config['batch_size'], _run.config['num_workers'], verbose=True)
    print("Training in batches of size %d.." % _run.config['batch_size'])

    # load model:
    model = load_model(_run.config['model'], num_classes=ds.num_classes)
    model.to(device)
    print('training model {}'.format(model.__class__.__name__))
    basename = _run.config['dataset'] + '_' + model.__class__.__name__

    # set loss function, set class weights:
    class_weights = [1/f for f in ds.trainset.class_ratio]
    class_weights = torch.Tensor(class_weights).to(device)
    print("class weights are:"); print(class_weights)
    criterion = load_loss(_run.config['loss'], weight=class_weights)

    # Select optimizer:
    lr = _run.config['learningrate']
    if _run.config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=_run.config['learningrate'], momentum=0.99)
    elif _run.config['optimizer'] == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=_run.config['learningrate'], betas=(0.9, 0.999))#, weight_decay=1e-5)
    else:
        sys.exit('ERROR: optimizer name does not exist..')
    print("Using %s optimizer, lr=%f.." % (_run.config['optimizer'], _run.config['learningrate']))

    # logs:
    log_dir = os.path.join(logs_dir, 'logs', _run.config['dataset'], str(_run.config['negs_set']), _run.config['model'], _run._id)
    writer = SummaryWriter(log_dir=log_dir)
    metric_dict = {'AUC/Best_Validation': 0}
    run_dict = {k: v.__repr__() if isinstance(v, list) else v for k, v in _run.config.items() if v is not None}
    sei = better_hparams(writer, hparam_dict=run_dict, metric_dict=metric_dict)
    print("log dir is: {}".format(log_dir))

    # Train & Evaluate:
    best_val_metric = 0.0
    with tqdm(total=_run.config['epochs']) as pbar_main:
        for epoch in range(1, _run.config['epochs'] + 1):
            pbar_main.update()

            if _run.config['subset'] is not None:  # update samples from train set
                trainloader, _ = sample_dataset(ds.trainset, ds.testset, _run.config['overfit'], _run.config['subset'],
                                                _run.config['batch_size'], _run.config['num_workers'], verbose=False)

            train_loss, train_acc, train_auc, train_misc = run_epoch(epoch, loader=trainloader, optimizer=optimizer,
                                                                     model=model, criterion=criterion, device=device, prefix='Training')

            if epoch == 1 or epoch % _run.config['evaluation_interval'] == 0:
                val_loss, val_acc, val_auc, val_misc = run_epoch(epoch, loader=valloader, optimizer=None, model=model,
                                                                 criterion=criterion, device=device, prefix='Validating')

            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            writer.add_scalar('AUC/Train', train_auc, epoch)
            writer.add_scalar('AUC/Validation', val_auc, epoch)

            # Update best:
            if val_auc > best_val_metric:
                filename = os.path.join(log_dir, 'weights_{}_best.pth'.format(basename))
                best_val_metric = val_auc
                save_model(filename, epoch, model, _run.config, train_loss, val_loss, best_val_metric, val_acc, hostname)
                writer.add_scalar('AUC/Best_Validation', best_val_metric, epoch)

    # Save & Close:
    filename = os.path.join(log_dir, 'weights_{}_ep_{}.pth'.format(basename, epoch))
    save_model(filename, epoch, model, _run.config, train_loss, val_loss, best_val_metric, val_acc, hostname)
    writer.file_writer.add_summary(sei)
    writer.close()
    print('Finished Training')

    # Eval:
    val_total_outputs_class, val_total_gt_labels, val_total_filenames = val_misc[0], val_misc[1], val_misc[2]
    filename = os.path.join(log_dir, 'misclassifications_{}_ep_{}'.format(basename, epoch))
    eval_model(val_total_outputs_class, val_total_gt_labels, val_total_filenames, filename)


