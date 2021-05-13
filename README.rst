What is ProteinBERT?
=============

ProteinBERT is a universal protein language model pretrained on ~106M proteins from the UniRef90 dataset. Through its Python API, the pretrained model can be fine-tuned on any protein-related task in a matter of minutes. Based on our experiments with a wide range of benchmarks, ProteinBERT usually achieves state-of-the-art performance. ProteinBERT is built on TenforFlow/Keras.

ProteinBERT's deep-learning architecture is inspired by BERT, but it contains several innovations such as its global-attention layers that grow only lineraly with sequence length (compared to self-attention's quadratic growth). As a result, the model can process protein sequences of almost any length, includng extremely long protein sequences (of over tens of thousands of amino acids).

The model takes protein sequences as inputs, and can also take protein GO annotations as additional inputs (to help the model infer about the function of the input protein and update its internal representations and outputs accordingly).
This package provides seamless access to a pretrained state that has been produced by training the model for 28 days over ~670M records (i.e. ~6.4 iterations over the entire training dataset of ~106M records). For users interested in pretraining the model from scratch, the package also includes scripts for that.


Installation
=============

Dependencies
------------

ProteinBERT requires Python 3.

Below are the Python packages required by ProteinBERT, which are automatically installed with it (and the versions of these packages that were tested with ProteinBERT 1.0.0):

* tensorflow (2.4.0)
* numpy (1.20.1)
* pandas (1.2.3)
* h5py (3.2.1)
* lxml (4.3.2)
* pyfaidx (0.5.8)


Install ProteinBERT
------------

Just run:

.. code-block:: sh

    pip install protein-bert
    
Alternatively, clone this repository and run:

.. code-block:: sh

    python setup.py install

