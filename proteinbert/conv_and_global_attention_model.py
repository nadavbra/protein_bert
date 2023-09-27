import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

class GlobalAttention(keras.layers.Layer):
    
    '''
    Recevies two inputs:
    1. A global representation (of some fixed dimension)
    2. A sequence (of any length, and some fixed dimension)
    The global representation is used to construct a global query that attends to all the positions in the sequence (independently
    for any of the heads).
    '''
    
    def __init__(self, n_heads, d_key, d_value, **kwargs):
        self.n_heads = n_heads
        self.d_key = d_key
        self.sqrt_d_key = np.sqrt(self.d_key)
        self.d_value = d_value
        self.d_output = n_heads * d_value
        super(GlobalAttention, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shapes):
        # input_shapes: (batch_size, d_global_input), (batch_size, length, d_seq_input)
        (batch_size, _), _ = input_shapes
        return (batch_size, self.d_output)

    def build(self, input_shapes):
        # input_shapes: (batch_size, d_global_input), (batch_size, length, d_seq_input)
        (_, self.d_global_input), (_, _, self.d_seq_input) = input_shapes
        # Wq: (n_heads, d_global_input, d_key)
        self.Wq = self.add_weight(name = 'Wq', shape = (self.n_heads, self.d_global_input, self.d_key), \
                initializer = 'glorot_uniform', trainable = True)
        # Wk: (n_heads, d_seq_input, d_key)
        self.Wk = self.add_weight(name = 'Wk', shape = (self.n_heads, self.d_seq_input, self.d_key), \
                initializer = 'glorot_uniform', trainable = True)
        # Wv: (n_heads, d_seq_input, d_value)
        self.Wv = self.add_weight(name = 'Wv', shape = (self.n_heads, self.d_seq_input, self.d_value), \
                initializer = 'glorot_uniform', trainable = True)
        super(GlobalAttention, self).build(input_shapes)

    def call(self, inputs):
    
        # X: (batch_size, d_global_input)
        # S: (batch_size, length, d_seq_input)
        X, S = inputs
        _, length, _ = K.int_shape(S)
    
        # (batch_size, n_heads, length, d_value)
        VS = K.permute_dimensions(keras.activations.gelu(K.dot(S, self.Wv)), (0, 2, 1, 3))
        # (batch_size * n_heads, length, d_value)
        VS_batched_heads = K.reshape(VS, (-1, length, self.d_value))
        
        Z_batched_heads = self.calculate_attention(inputs)
        # (batch_size * n_heads, d_value)
        Y_batched_heads = K.batch_dot(Z_batched_heads, VS_batched_heads)
        # (batch_size, n_heads * d_value)
        Y = K.reshape(Y_batched_heads, (-1, self.d_output))
        
        return Y
        
    def calculate_attention(self, inputs):
    
        # X: (batch_size, d_global_input)
        # S: (batch_size, length, d_seq_input)
        X, S = inputs
        _, length, _ = K.int_shape(S)
                
        # (batch_size, n_heads, d_key)
        QX = K.tanh(K.dot(X, self.Wq))
        # (batch_size * n_heads, d_key)
        QX_batched_heads = K.reshape(QX, (-1, self.d_key))
        
        # (batch_size, n_heads, d_key, length)
        KS = K.permute_dimensions(K.tanh(K.dot(S, self.Wk)), (0, 2, 3, 1))
        # (batch_size * n_heads, d_key, length)
        KS_batched_heads = K.reshape(KS, (-1, self.d_key, length))
                
        # (batch_size * n_heads, length)
        Z_batched_heads = K.softmax(K.batch_dot(QX_batched_heads, KS_batched_heads) / self.sqrt_d_key)
        return Z_batched_heads
    
def create_model(seq_len, vocab_size, n_annotations, d_hidden_seq = 128, d_hidden_global = 512, n_blocks = 6, n_heads = 4, \
         d_key = 64, conv_kernel_size = 9, wide_conv_dilation_rate = 5, activation = 'gelu'):
    
    '''
    seq_len is required to create the model, but all the weights are independent of the length and can be re-used with
    different lengths.
    '''
    
    assert d_hidden_global % n_heads == 0
    d_value = d_hidden_global // n_heads
    
    input_seq = keras.layers.Input(shape = (seq_len,), dtype = np.int32, name = 'input-seq')
    input_annotations = keras.layers.Input(shape = (n_annotations,), dtype = np.float32, name = 'input-annotations')
    
    hidden_seq = keras.layers.Embedding(vocab_size, d_hidden_seq, name = 'embedding-seq-input')(input_seq)
    hidden_global = keras.layers.Dense(d_hidden_global, activation = activation, name = 'dense-global-input')(input_annotations)
    
    for block_index in range(1, n_blocks + 1):
        
        seqed_global = keras.layers.Dense(d_hidden_seq, activation = activation, name = 'global-to-seq-dense-block%d' % block_index)(hidden_global)
        seqed_global = keras.layers.Reshape((1, d_hidden_seq), name = 'global-to-seq-reshape-block%d' % block_index)(seqed_global)
        
        narrow_conv_seq = keras.layers.Conv1D(filters = d_hidden_seq, kernel_size = conv_kernel_size, strides = 1, \
                padding = 'same', dilation_rate = 1, activation = activation, name = 'narrow-conv-block%d' % block_index)(hidden_seq)
        wide_conv_seq = keras.layers.Conv1D(filters = d_hidden_seq, kernel_size = conv_kernel_size, strides = 1, \
                padding = 'same', dilation_rate = wide_conv_dilation_rate, activation = activation, name = 'wide-conv-block%d' % \
                block_index)(hidden_seq)
        
        hidden_seq = keras.layers.Add(name = 'seq-merge1-block%d' % block_index)([hidden_seq, seqed_global, narrow_conv_seq, wide_conv_seq])
        hidden_seq = keras.layers.LayerNormalization(name = 'seq-merge1-norm-block%d' % block_index)(hidden_seq)
        
        dense_seq = keras.layers.Dense(d_hidden_seq, activation = activation, name = 'seq-dense-block%d' % block_index)(hidden_seq)
        hidden_seq = keras.layers.Add(name = 'seq-merge2-block%d' % block_index)([hidden_seq, dense_seq])
        hidden_seq = keras.layers.LayerNormalization(name = 'seq-merge2-norm-block%d' % block_index)(hidden_seq)
        
        dense_global = keras.layers.Dense(d_hidden_global, activation = activation, name = 'global-dense1-block%d' % block_index)(hidden_global)
        attention = GlobalAttention(n_heads, d_key, d_value, name = 'global-attention-block%d' % block_index)([hidden_global, hidden_seq])
        hidden_global = keras.layers.Add(name = 'global-merge1-block%d' % block_index)([hidden_global, dense_global, attention])
        hidden_global = keras.layers.LayerNormalization(name = 'global-merge1-norm-block%d' % block_index)(hidden_global)
        
        dense_global = keras.layers.Dense(d_hidden_global, activation = activation, name = 'global-dense2-block%d' % block_index)(hidden_global)
        hidden_global = keras.layers.Add(name = 'global-merge2-block%d' % block_index)([hidden_global, dense_global])
        hidden_global = keras.layers.LayerNormalization(name = 'global-merge2-norm-block%d' % block_index)(hidden_global)
        
    output_seq = keras.layers.Dense(vocab_size, activation = 'softmax', name = 'output-seq')(hidden_seq)
    output_annotations = keras.layers.Dense(n_annotations, activation = 'sigmoid', name = 'output-annotations')(hidden_global)

    return keras.models.Model(inputs = [input_seq, input_annotations], outputs = [output_seq, output_annotations])
    
def get_model_with_hidden_layers_as_outputs(model):
    
    _, seq_len, _ = model.outputs[0].shape
    
    seq_layers = [layer.output for layer in model.layers if len(layer.output.shape) == 3 and \
            tuple(layer.output.shape)[:2] == (None, seq_len) and (layer.name in ['input-seq-encoding', 'dense-seq-input', 'output-seq'] or \
            isinstance(layer, keras.layers.LayerNormalization))]
    global_layers = [layer.output for layer in model.layers if len(layer.output.shape) == 2 and (layer.name in ['input_annotations', \
            'dense-global-input', 'output-annotations'] or isinstance(layer, keras.layers.LayerNormalization))]
    
    concatenated_seq_output = keras.layers.Concatenate(name = 'all-seq-layers')(seq_layers)
    concatenated_global_output = keras.layers.Concatenate(name = 'all-global-layers')(global_layers)
    
    return keras.models.Model(inputs = model.inputs, outputs = [concatenated_seq_output, concatenated_global_output])
    
