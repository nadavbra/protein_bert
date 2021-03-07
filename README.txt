> cd ~/tmp/new_proteinbert_dataset
> wget https://www.biofunctionprediction.org/cafa-targets/cafa4ontologies.zip
> mkdir cafa4ontologies
> unzip cafa4ontologies.zip -d cafa4ontologies/
> wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.xml.gz
> wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
> ... uncompress fasta.gz

> setenv PYTHONPATH ~/github_projects/protein_bert
> nohup ~/github_projects/protein_bert/bin/create_uniref_db --uniref-xml-gz-file=./uniref90.xml.gz --go-annotations-meta-file=./cafa4ontologies/go.txt --output-sqlite-file=./uniref_proteins_and_annotations.db --output-go-annotations-meta-csv-file=./go_annotations.csv >&! ./log_create_uniref_db.txt &

> ~/github_projects/protein_bert/bin/create_uniref_h5_dataset --protein-annotations-sqlite-db-file=./uniref_proteins_and_annotations.db --protein-fasta-file=./uniref90.fasta --go-annotations-meta-csv-file=./go_annotations_tmp.csv --output-h5-dataset-file=./dataset.h5 --min-records-to-keep-annotation=100
--records-limit=100000


> ~/github_projects/protein_bert/bin/set_h5_testset --h5-dataset-file=./dataset.h5 --uniprot-ids-file=~/tmp/new_proteinbert_dataset/uniref_90_seqs_matching_test_set_seqs.txt


