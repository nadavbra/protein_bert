#! /usr/bin/env python3

import os
import re
import json
import argparse
from datetime import timedelta

from tensorflow import keras

from proteinbert.shared_utils.util import log, load_object, get_parser_file_type, get_parser_directory_type
from proteinbert.pretraining import run_pretraining, EpochGenerator, AutoSaveManager

class FixedReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
    def on_train_begin(self, logs = None):
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Runs the pretraining of a ProteinBERT model (from scratch or from an existing state) over a dataset of ' + \
            'protein sequences and annotations.')
    parser.add_argument('--dataset-file', dest = 'dataset_file', metavar = '/path/to/dataset.h5', type = get_parser_file_type(parser, must_exist = True), \
            required = True, help = 'The dataset h5 file with the protein sequences and annotations to train on.')
    parser.add_argument('--autosave-dir', dest = 'autosave_dir', metavar = '/path/to/autosave_dir/', type = get_parser_directory_type(parser, create_if_not_exists = True), \
            required = True, help = 'The directory to save the model state into throughout the training process.')
    parser.add_argument('--resume-from', dest = 'resume_from', metavar = '/path/to/epoch_XXX_sample_XXX.pkl', type = get_parser_file_type(parser, must_exist = True), \
            default = None, help = 'Use the provided saved model state to resume training from.')
    parser.add_argument('--epochs', dest = 'epochs', metavar = 'n', type = int, default = None, help = 'Run for the given number of epochs before exiting ' + \
            '(by default will continue for ever, until interrupted).')
    parser.add_argument('--epochs-per-autosave', dest = 'epochs_per_autosave', metavar = '10', type = int, default = 10, help = 'The number of epochs per autosaving ' + \
            'of the model state.')
    parser.add_argument('--keep-every-autosaves', dest = 'keep_every_autosaves', metavar = '25', type = int, default = 25, help = 'Keep every Nth autosave, for this ' + \
            'number N (and delete the rest, once a more updated autosave is made).')
    parser.add_argument('--batches-per-epoch', dest = 'batches_per_epoch', metavar = '100', type = int, default = 100, help = 'The number of batches to define each epoch.')
    parser.add_argument('--minutes-per-episode', dest = 'minutes_per_episode', metavar = '15.0', type = float, default = 15.0, help = 'The time span (in minutes) to wait ' + \
            'before changing to a new episode (once the current epoch is finished). Each episode is defined by a different sequence length of the model.')
    parser.add_argument('--seq-noise-prob', dest = 'seq_noise_prob', metavar = '0.05', type = float, default = 0.05, help = 'The probability for noising each token ' + \
            'in the input sequences and replacing it with a random token (letting the model predict the original tokens).')
    parser.add_argument('--no-input-annotations-prob', dest = 'no_input_annotations_prob', metavar = '0.5', type = float, default = 0.5, help = 'The probability for not ' + \
            'providing any annotations as input alongside the sequence (meaning that the model would have to predict all the annotations from the sequence alone).')
    parser.add_argument('--remove-annotation-prob', dest = 'remove_annotation_prob', metavar = '0.25', type = float, default = 0.25, help = 'The probability for noising ' + \
            'the input annotations by removing a true annotation (letting the model recover it).')
    parser.add_argument('--add-annotation-prob', dest = 'add_annotation_prob', metavar = '1e-04', type = float, default = 1e-04, help = 'The probability for noising ' + \
            'the input annotations by adding a negative annotation that is not really associated with the protein (letting the model learn to ignore it).')
    parser.add_argument('--annotations-loss-weight', dest = 'annotations_loss_weight', metavar = '1.0', type = float, default = 1.0, help = 'A scaling factor for ' + \
            'the loss for the annotations (relative to the loss for the sequence).')
    parser.add_argument('--lr', dest = 'lr', metavar = '2e-04', type = float, default = 2e-04, help = 'Learning rate.')
    parser.add_argument('--no-lr-reduction', dest = 'no_lr_reduction', action = 'store_true', help = 'Provide this flag to not use ReduceLROnPlateau (which is used by default).')
    parser.add_argument('--min-lr', dest = 'min_lr', metavar = '5e-05', type = float, default = 5e-05, help = 'ReduceLROnPlateau.min_lr')
    parser.add_argument('--lr-reduction-patience', dest = 'lr_reduction_patience', metavar = '20', type = int, default = 20, help = 'ReduceLROnPlateau.patience')
    parser.add_argument('--lr-reduction-factor', dest = 'lr_reduction_factor', metavar = '0.8', type = float, default = 0.8, help = 'ReduceLROnPlateau.factor')
    parser.add_argument('--optimizer-class', dest = 'optimizer_class', metavar = 'tensorflow_addons.optimizers.LAMB', type = str, \
            default = 'tensorflow_addons.optimizers.LAMB', help = 'The optimizer class to use.')
    parser.add_argument('--create-model-function', dest = 'create_model_function', metavar = 'proteinbert.conv_and_global_attention_model.create_model', type = str, \
            default = 'proteinbert.conv_and_global_attention_model.create_model', help = 'The function to create the model.')
    parser.add_argument('--create-model-kwargs-file', dest = 'create_model_kwargs_file', metavar = '/path/to/kwargs.json', \
            type = get_parser_file_type(parser, must_exist = True), default = None, help = 'A JSON file specifying kwargs for creating the model.')
    parser.add_argument('--load-chunk-size', dest = 'load_chunk_size', metavar = '100000', type = int, default = 100000, help = 'The number of records to load into the ' + \
            'memory whenever new records are needed.')
    args = parser.parse_args()
    
    epoch_generator = EpochGenerator(n_batches_per_epoch = args.batches_per_epoch, p_seq_noise = args.seq_noise_prob, p_no_input_annot = args.no_input_annotations_prob, \
            p_annot_noise_positive = args.remove_annotation_prob, p_annot_noise_negative = args.add_annotation_prob, load_chunk_size = args.load_chunk_size, \
            min_time_per_episode = timedelta(minutes = args.minutes_per_episode))
    autosave_manager = AutoSaveManager(args.autosave_dir, every_epochs_to_save = args.epochs_per_autosave, every_saves_to_keep = args.keep_every_autosaves)
    
    create_model_function = load_object(args.create_model_function)
    optimizer_class = load_object(args.optimizer_class)
    
    if args.create_model_kwargs_file is None:
        create_model_kwargs = {}
    else:
        with open(args.create_model_kwargs_file, 'r') as f:
            create_model_kwargs = json.load(f)
    
    if args.resume_from is None:
        load_weights_dir = None
        resume_from = None
    else:
        
        load_weights_dir = os.path.dirname(args.resume_from)
        raw_resume_from = os.path.basename(args.resume_from)
        
        try:
            resume_from, = re.findall(r'epoch_(\d+)_sample_(\d+)\.pkl', raw_resume_from)
            resume_from = tuple(map(int, resume_from))
        except:
            raise parser.error('Invalid file name to resume training from: %s' % raw_resume_from)
            
    if args.no_lr_reduction:
        fit_callbacks = []
    else:
        lr_adjusting_callback = FixedReduceLROnPlateau(patience = args.lr_reduction_patience, factor = args.lr_reduction_factor, min_lr = args.min_lr, \
                monitor = 'loss', verbose = 1)
        fit_callbacks = [lr_adjusting_callback]
    
    run_pretraining(create_model_function, epoch_generator, args.dataset_file, create_model_kwargs = create_model_kwargs, optimizer_class = optimizer_class, lr = args.lr, \
            annots_loss_weight = args.annotations_loss_weight, autosave_manager = autosave_manager, weights_dir = load_weights_dir, resume_from = resume_from, \
            n_epochs = args.epochs, fit_callbacks = fit_callbacks)
    log('Done.')
