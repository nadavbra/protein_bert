What is ProteinBERT?
=============

ProteinBERT is a protein language model pretrained on ~106M proteins from UniRef90. The pretrained model can be fine-tuned on any protein-related task in a matter of minutes. ProteinBERT achieves state-of-the-art performance on a wide range of benchmarks. ProteinBERT is built on Keras/TensorFlow.

ProteinBERT's deep-learning architecture is inspired by BERT, but contains several innovations such as  global-attention layers that have linear complexity for sequence length (compared to self-attention's quadratic/n^2 growth). As a result, the model can process protein sequences of almost any length, including extremely long protein sequences (of over tens of thousands of amino acids).

The model takes protein sequences as inputs, and can also take protein GO annotations as additional inputs (to help the model infer about the function of the input protein and update its internal representations and outputs accordingly).
This package provides access to a pretrained model produced by training for 28 days over ~670M records (~6.4 epochs over the entire UniRef90 training dataset of ~106M proteins). The package also includes scripts for pretraining the model from scratch and extracting the relevant data.


Getting started with pretrained ProteinBERT embeddings
=============
Here's a quick code snippet for getting embeddings at the whole sequence (protein) level - you can use these for downstream tasks as extracted features with other ML models, clustering, KNN, etc'. (You can also get local/position level embeddings, and fine tune the ProteinBERT model itself on your task).

```
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))
encoded_x = input_encoder.encode_X(seqs, seq_len)
local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
# ... use these as features for other tasks, based on local_representations, global_representations
```
Have a look at the notebook used to finetune the model on a large set of diverse tasks and benchmarks for more usage examples:
[ProteinBERT demo](https://github.com/nadavbra/protein_bert/blob/master/ProteinBERT%20demo.ipynb).

You can also download  directly from Huggingface as a Keras model: https://huggingface.co/GrimSqueaker/proteinBERT


Installation
=============

Dependencies
------------

ProteinBERT requires Python 3.

Below are the Python packages required by ProteinBERT, which are automatically installed with it (and the versions of these packages that were tested with ProteinBERT 1.0.0):

* tensorflow (2.4.0)
* tensorflow_addons (0.12.1)
* numpy (1.20.1)
* pandas (1.2.3)
* h5py (3.2.1)
* lxml (4.3.2)
* pyfaidx (0.5.8)


Install ProteinBERT
------------

Clone this repository and run:

```sh
git submodule init
git submodule update
python setup.py install
```    
    
Using ProteinBERT
=============

Fine-tuning ProteinBERT is easy. You can see working examples [in this notebook](https://github.com/nadavbra/protein_bert/blob/master/ProteinBERT%20demo.ipynb).

You can download the pretrained model & weights from Zenodo at https://zenodo.org/records/10371965 or from GitHub at https://github.com/nadavbra/proteinbert_data_files/blob/master/epoch_92400_sample_23500000.pkl

The model is also available on Huggingface: https://huggingface.co/GrimSqueaker/proteinBERT
    
Pretraining ProteinBERT from scratch
=============

If, instead of using the existing model weights, you would like to train from scratch, then follow the steps below. We warn that this is a long process (we pretrained the current model for a whole month), and it also requires a lot of storage (>1TB).

Step 1: Create the UniRef dataset
------------

ProteinBERT is pretrained on a dataset derived from UniRef90. Follow these steps to produce this dataset:

1. First, choose a working directory with sufficient (>1TB) free storage.

```sh    
cd /some/workdir
```

2. Download the metadata of GO from CAFA and extract it.

```sh
wget https://www.biofunctionprediction.org/cafa-targets/cafa4ontologies.zip
mkdir cafa4ontologies
unzip cafa4ontologies.zip -d cafa4ontologies/
```
    
3. Download UniRef90, as both XML and FASTA.

```sh
wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.xml.gz
wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
gunzip uniref90.fasta.gz
```
    
4. Use the *create_uniref_db* script provided by ProteinBERT to extract the GO annotations associated with UniRef's records into an SQLite database (and a CSV file with the metadata of these GO annotations). Since this is a long process (which can take up to a few days), it is recommended to run this in the background (e.g. using *nohup*).
    
```sh
nohup create_uniref_db --uniref-xml-gz-file=./uniref90.xml.gz --go-annotations-meta-file=./cafa4ontologies/go.txt --output-sqlite-file=./uniref_proteins_and_annotations.db --output-go-annotations-meta-csv-file=./go_annotations.csv >&! ./log_create_uniref_db.txt &
```
    
5. Create the final dataset (in the H5 format) by merging the database of GO annotations with the protein sequences using the *create_uniref_h5_dataset* script provided by ProteinBERT. This is also a long process that should be let to run in the background.

```sh    
nohup create_uniref_h5_dataset --protein-annotations-sqlite-db-file=./uniref_proteins_and_annotations.db --protein-fasta-file=./uniref90.fasta --go-annotations-meta-csv-file=./go_annotations.csv --output-h5-dataset-file=./dataset.h5 --min-records-to-keep-annotation=100 >&! ./log_create_uniref_h5_dataset.txt &
```
    
6. Finally, use ProteinBERT's *set_h5_testset* script to designate which of the dataset records will be considered part of the test set (so that their GO annotations are not used during pretraining). If you are planning to evaluate your model on certain downstream benchmarks, it is recommended that any UniRef record similar to a test-set protein in these benchmark will be considered part of the pretraining's test set. You can use BLAST to find all of these UniRef records and provide them to *set_h5_testset* through the flag ``--uniprot-ids-file=./uniref_90_seqs_matching_test_set_seqs.txt``, where the provided text file contains the UniProt IDs of the relevant records, one per line (e.g. *A0A009EXK6_ACIBA*).

```sh
set_h5_testset --h5-dataset-file=./dataset.h5
```
    
    
Step 2: Pretrain ProteinBERT on the UniRef dataset
------------

Once you have the dataset ready, the *pretrain_proteinbert* script will train a ProteinBERT model on that dataset.

Basic use of the pretraining script looks as follows:

```sh
mkdir -p ~/proteinbert_models/new
nohup pretrain_proteinbert --dataset-file=./dataset.h5 --autosave-dir=~/proteinbert_models/new >&! ~/proteinbert_models/log_new_pretraining.txt &
```
    
By running that, ProteinBERT will continue to train indefinitely. Therefore, make sure to run it in the background using *nohup* or other options. Every given number of epochs (determined as 100 batches) the model state will be automatically saved into the specified autosave directory. If this process is interrupted and you wish to resume pretraining
from a given snapshot (e.g. the most up-to-date state file within the autosave dir) use the ``--resume-from`` flag (provide it the state file that you wish to resume from).

*pretrain_proteinbert* has MANY options and hyper-parameters that are worth checking out:

```sh
pretrain_proteinbert --help
```    
    
Step 3: Use your pretrained model state when fine-tuning ProteinBERT
------------

Normally the function *load_pretrained_model* is used to load the existing pretrained model state. If you wish to load your own pretrained model state instead, then use the *load_pretrained_model_from_dump* function instead.

Downloading the supervised benchmarks
=======
You can download the evaluation benchmarks from https://github.com/nadavbra/proteinbert_data_files/tree/master/protein_benchmarks.
    
Other implementations:
=======
An unofficial PyTorch implementation is also available: https://github.com/lucidrains/protein-bert-pytorch

License
=======
ProteinBERT is a free open-source project available under the `MIT License <https://en.wikipedia.org/wiki/MIT_License>`_.
 
## Citation <a name="citations"></a>
=======

If you use ProteinBERT, we ask that you cite our paper:
``` 
Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. & Linial, M. 
ProteinBERT: A universal deep-learning model of protein sequence and function. 
Bioinformatics (2022). https://doi.org/10.1093/bioinformatics/btac020
```

```bibtex
@article{10.1093/bioinformatics/btac020,
    author = {Brandes, Nadav and Ofer, Dan and Peleg, Yam and Rappoport, Nadav and Linial, Michal},
    title = "{ProteinBERT: a universal deep-learning model of protein sequence and function}",
    journal = {Bioinformatics},
    volume = {38},
    number = {8},
    pages = {2102-2110},
    year = {2022},
    month = {02},
    abstract = "{Self-supervised deep language modeling has shown unprecedented success across natural language tasks, and has recently been repurposed to biological sequences. However, existing models and pretraining methods are designed and optimized for text analysis. We introduce ProteinBERT, a deep language model specifically designed for proteins. Our pretraining scheme combines language modeling with a novel task of Gene Ontology (GO) annotation prediction. We introduce novel architectural elements that make the model highly efficient and flexible to long sequences. The architecture of ProteinBERT consists of both local and global representations, allowing end-to-end processing of these types of inputs and outputs. ProteinBERT obtains near state-of-the-art performance, and sometimes exceeds it, on multiple benchmarks covering diverse protein properties (including protein structure, post-translational modifications and biophysical attributes), despite using a far smaller and faster model than competing deep-learning methods. Overall, ProteinBERT provides an efficient framework for rapidly training protein predictors, even with limited labeled data.Code and pretrained model weights are available at https://github.com/nadavbra/protein\_bert.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac020},
    url = {https://doi.org/10.1093/bioinformatics/btac020},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/8/2102/45474534/btac020.pdf},
}
```
