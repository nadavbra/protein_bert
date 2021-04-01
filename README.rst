Temporary instructions
=============

1. There is no yet a proper setup.py/pip install to this project, so simply clone the project and add the proteinbert subdirectory to your python-path (e.g. by modifying the PYTHONPATH virtualenv, or by using sys.path.append).
2. Download the `demo Jupyter notebook <https://github.com/nadavbra/protein_bert/blob/master/ProteinBERT%20demo.ipynb>`_ to learn how to fine-tune a pre-trained proteinbert model on a new protein dataset.

TODO
=============

* Update ProteinBERT demo.ipynb
* Create setup.py and pip install
* Write proper documentation (including installation instructions and dependencies, usage, some background about the package, and how to cite it).
    - Notice that the dependencies include tensorflow_addons
    - Include instructions on how to create the dataset from scratch (see below)
    
XXX Creating the pretraining UniRef dataset
=============

> activate_my_python
> setenv PYTHONPATH ~/github_projects/protein_bert
> cd ...

> wget https://www.biofunctionprediction.org/cafa-targets/cafa4ontologies.zip
> mkdir cafa4ontologies
> unzip cafa4ontologies.zip -d cafa4ontologies/

> wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.xml.gz
> wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
> gunzip uniref90.fasta.gz

> nohup ~/github_projects/protein_bert/bin/create_uniref_db --uniref-xml-gz-file=./uniref90.xml.gz --go-annotations-meta-file=./cafa4ontologies/go.txt --output-sqlite-file=./uniref_proteins_and_annotations.db --output-go-annotations-meta-csv-file=./go_annotations.csv >&! ./log_create_uniref_db.txt &

> nohup ~/github_projects/protein_bert/bin/create_uniref_h5_dataset --protein-annotations-sqlite-db-file=./uniref_proteins_and_annotations.db --protein-fasta-file=./uniref90.fasta --go-annotations-meta-csv-file=./go_annotations.csv --output-h5-dataset-file=./dataset.h5 --min-records-to-keep-annotation=100 >&! ./log_create_uniref_h5_dataset.txt

> ~/github_projects/protein_bert/bin/set_h5_testset --h5-dataset-file=./dataset.h5 --uniprot-ids-file=./uniref_90_seqs_matching_test_set_seqs.txt


XXX Pretraining the model
=============

> activate_my_python
> setenv PYTHONPATH ~/github_projects/protein_bert
> module load cuda/11.0
> module load cudnn/8.0.2
> nohup ~/github_projects/protein_bert/bin/pretrain_proteinbert --dataset-file=~/tmp/new_proteinbert_dataset/dataset.h5 --autosave-dir=~/proteinbert_models/new >&! ~/proteinbert_models/log_new_pretraining.txt &

# --resume-from=.../epoch_92400_sample_23500000.pkl

