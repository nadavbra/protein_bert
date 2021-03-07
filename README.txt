> cd ~/tmp/new_proteinbert_dataset
> wget https://www.biofunctionprediction.org/cafa-targets/cafa4ontologies.zip
> mkdir cafa4ontologies
> unzip cafa4ontologies.zip -d cafa4ontologies/
> wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.xml.gz

> setenv PYTHONPATH ~/github_projects/protein_bert
> nohup ~/github_projects/protein_bert/bin/create_uniref_db --uniref-xml-gz-file=./uniref90.xml.gz --go-annotations-meta-file=./cafa4ontologies/go.txt --output-sqlite-file=./uniref_proteins_and_annotations.db --output-go-annotations-meta-csv-file=./go_annotations.csv >&! ./log_create_uniref_db.txt &
