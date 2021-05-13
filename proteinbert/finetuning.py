import numpy as np
import pandas as pd

from .shared_utils.util import log
from .tokenization import ADDED_TOKENS_PER_SEQ

class OutputType:
    
    def __init__(self, is_seq, output_type):
        self.is_seq = is_seq
        self.output_type = output_type
        self.is_numeric = (output_type == 'numeric')
        self.is_binary = (output_type == 'binary')
        self.is_categorical = (output_type == 'categorical')
        
    def __str__(self):
        if self.is_seq:
            return '%s sequence' % self.output_type
        else:
            return 'global %s' % self.output_type
            
class OutputSpec:

    def __init__(self, output_type, unique_labels = None):
        
        if output_type.is_numeric:
            assert unique_labels is None
        elif output_type.is_binary:
            if unique_labels is None:
                unique_labels = [0, 1]
            else:
                assert unique_labels == [0, 1]
        elif output_type.is_categorical:
            assert unique_labels is not None
        else:
            raise ValueError('Unexpected output type: %s' % output_type)
        
        self.output_type = output_type
        self.unique_labels = unique_labels
        
        if unique_labels is not None:
            self.n_unique_labels = len(unique_labels)
            
def finetune(model_generator, input_encoder, output_spec, train_seqs, train_raw_Y, valid_seqs = None, valid_raw_Y = None, seq_len = 512, batch_size = 32, \
        max_epochs_per_stage = 40, lr = None, begin_with_frozen_pretrained_layers = True, lr_with_frozen_pretrained_layers = None, n_final_epochs = 1, \
        final_seq_len = 1024, final_lr = None, callbacks = []):
        
    encoded_train_set, encoded_valid_set = encode_train_and_valid_sets(train_seqs, train_raw_Y, valid_seqs, valid_raw_Y, input_encoder, output_spec, seq_len)
        
    if begin_with_frozen_pretrained_layers:
        log('Training with frozen pretrained layers...')
        model_generator.train(encoded_train_set, encoded_valid_set, seq_len, batch_size, max_epochs_per_stage, lr = lr_with_frozen_pretrained_layers, \
                callbacks = callbacks, freeze_pretrained_layers = True)
     
    log('Training the entire fine-tuned model...')
    model_generator.train(encoded_train_set, encoded_valid_set, seq_len, batch_size, max_epochs_per_stage, lr = lr, callbacks = callbacks, \
            freeze_pretrained_layers = False)
                
    if n_final_epochs > 0:
        log('Training on final epochs of sequence length %d...' % final_seq_len)
        final_batch_size = max(int(batch_size / (final_seq_len / seq_len)), 1)
        encoded_train_set, encoded_valid_set = encode_train_and_valid_sets(train_seqs, train_raw_Y, valid_seqs, valid_raw_Y, input_encoder, output_spec, final_seq_len)
        model_generator.train(encoded_train_set, encoded_valid_set, final_seq_len, final_batch_size, n_final_epochs, lr = final_lr, callbacks = callbacks, \
                freeze_pretrained_layers = False)
                
    model_generator.optimizer_weights = None

def evaluate_by_len(model_generator, input_encoder, output_spec, seqs, raw_Y, start_seq_len = 512, start_batch_size = 32, increase_factor = 2):
    
    assert model_generator.optimizer_weights is None
    
    dataset = pd.DataFrame({'seq': seqs, 'raw_y': raw_Y})
        
    results = []
    results_names = []
    y_trues = []
    y_preds = []
    
    for len_matching_dataset, seq_len, batch_size in split_dataset_by_len(dataset, start_seq_len = start_seq_len, start_batch_size = start_batch_size, \
            increase_factor = increase_factor):

        X, y_true, sample_weights = encode_dataset(len_matching_dataset['seq'], len_matching_dataset['raw_y'], input_encoder, output_spec, \
                seq_len = seq_len, needs_filtering = False)
        
        assert set(np.unique(sample_weights)) <= {0.0, 1.0}
        y_mask = (sample_weights == 1)
        
        model = model_generator.create_model(seq_len)
        y_pred = model.predict(X, batch_size = batch_size)
        
        y_true = y_true[y_mask].flatten()
        y_pred = y_pred[y_mask]
        
        if output_spec.output_type.is_categorical:
            y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        else:
            y_pred = y_pred.flatten()
        
        results.append(get_evaluation_results(y_true, y_pred, output_spec))
        results_names.append(seq_len)
        
        y_trues.append(y_true)
        y_preds.append(y_pred)
        
    y_true = np.concatenate(y_trues, axis = 0)
    y_pred = np.concatenate(y_preds, axis = 0)
    all_results, confusion_matrix = get_evaluation_results(y_true, y_pred, output_spec, return_confusion_matrix = True)
    results.append(all_results)
    results_names.append('All')
    
    results = pd.DataFrame(results, index = results_names)
    results.index.name = 'Model seq len'
    
    return results, confusion_matrix

def get_evaluation_results(y_true, y_pred, output_spec, return_confusion_matrix = False):

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
            
    results = {}
    results['# records'] = len(y_true)
            
    if output_spec.output_type.is_numeric:
        results['Spearman\'s rank correlation'] = spearmanr(y_true, y_pred)[0]
        confusion_matrix = None
    else:
    
        str_unique_labels = list(map(str, output_spec.unique_labels))
        
        if output_spec.output_type.is_binary:
            
            y_pred_classes = (y_pred >= 0.5)
            
            if len(np.unique(y_true)) == 2:
                results['AUC'] = roc_auc_score(y_true, y_pred)
            else:
                results['AUC'] = np.nan
        elif output_spec.output_type.is_categorical:
            y_pred_classes = y_pred.argmax(axis = -1)
            results['Accuracy'] = accuracy_score(y_true, y_pred_classes)
        else:
            raise ValueError('Unexpected output type: %s' % output_spec.output_type)
                    
        confusion_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred_classes, labels = np.arange(output_spec.n_unique_labels)), index = str_unique_labels, \
                    columns = str_unique_labels)
         
    if return_confusion_matrix:
        return results, confusion_matrix
    else:
        return results
        
def encode_train_and_valid_sets(train_seqs, train_raw_Y, valid_seqs, valid_raw_Y, input_encoder, output_spec, seq_len):
    
    encoded_train_set = encode_dataset(train_seqs, train_raw_Y, input_encoder, output_spec, seq_len = seq_len, needs_filtering = True, \
            dataset_name = 'Training set')
    
    if valid_seqs is None and valid_raw_Y is None:
        encoded_valid_set = None
    else:
        encoded_valid_set = encode_dataset(valid_seqs, valid_raw_Y, input_encoder, output_spec, seq_len = seq_len, needs_filtering = True, \
                dataset_name = 'Validation set')

    return encoded_train_set, encoded_valid_set
        
def encode_dataset(seqs, raw_Y, input_encoder, output_spec, seq_len = 512, needs_filtering = True, dataset_name = 'Dataset', verbose = True):
    
    if needs_filtering:
        dataset = pd.DataFrame({'seq': seqs, 'raw_Y': raw_Y})
        dataset = filter_dataset_by_len(dataset, seq_len = seq_len, dataset_name = dataset_name, verbose = verbose)
        seqs = dataset['seq']
        raw_Y = dataset['raw_Y']
    
    X = input_encoder.encode_X(seqs, seq_len)
    Y, sample_weigths = encode_Y(raw_Y, output_spec, seq_len = seq_len)
    return X, Y, sample_weigths

def encode_Y(raw_Y, output_spec, seq_len = 512):
    if output_spec.output_type.is_seq:
        return encode_seq_Y(raw_Y, seq_len, output_spec.output_type.is_binary, output_spec.unique_labels)
    elif output_spec.output_type.is_categorical:
        return encode_categorical_Y(raw_Y, output_spec.unique_labels), np.ones(len(raw_Y))
    elif output_spec.output_type.is_numeric or output_spec.output_type.is_binary:
        return raw_Y.values.astype(float), np.ones(len(raw_Y))
    else:
        raise ValueError('Unexpected output type: %s' % output_spec.output_type)

def encode_seq_Y(seqs, seq_len, is_binary, unique_labels):

    label_to_index = {str(label): i for i, label in enumerate(unique_labels)}

    Y = np.zeros((len(seqs), seq_len), dtype = int)
    sample_weigths = np.zeros((len(seqs), seq_len))
    
    for i, seq in enumerate(seqs):
        
        for j, label in enumerate(seq):
            # +1 to account for the <START> token at the beginning.
            Y[i, j + 1] = label_to_index[label]
            
        sample_weigths[i, 1:(len(seq) + 1)] = 1
        
    if is_binary:
        Y = np.expand_dims(Y, axis = -1)
        sample_weigths = np.expand_dims(sample_weigths, axis = -1)
    
    return Y, sample_weigths
    
def encode_categorical_Y(labels, unique_labels):
    
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    Y = np.zeros(len(labels), dtype = int)
    
    for i, label in enumerate(labels):
        Y[i] = label_to_index[label]
        
    return Y
    
def filter_dataset_by_len(dataset, seq_len = 512, seq_col_name = 'seq', dataset_name = 'Dataset', verbose = True):
    
    max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ
    filtered_dataset = dataset[dataset[seq_col_name].str.len() <= max_allowed_input_seq_len]
    n_removed_records = len(dataset) - len(filtered_dataset)
    
    if verbose:
        log('%s: Filtered out %d of %d (%.1f%%) records of lengths exceeding %d.' % (dataset_name, n_removed_records, len(dataset), 100 * n_removed_records / len(dataset), \
                max_allowed_input_seq_len))
    
    return filtered_dataset
    
def split_dataset_by_len(dataset, seq_col_name = 'seq', start_seq_len = 512, start_batch_size = 32, increase_factor = 2):

    seq_len = start_seq_len
    batch_size = start_batch_size
    
    while len(dataset) > 0:
        max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ
        len_mask = (dataset[seq_col_name].str.len() <= max_allowed_input_seq_len)
        len_matching_dataset = dataset[len_mask]
        yield len_matching_dataset, seq_len, batch_size
        dataset = dataset[~len_mask]
        seq_len *= increase_factor
        batch_size = max(batch_size // increase_factor, 1)
