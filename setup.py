from setuptools import setup

setup(
    name = 'protein-bert',
    version = '1.0.1',
    description = 'A BERT-like deep language model for protein sequences.',
    url = 'https://github.com/nadavbra/protein_bert',
    author = 'Nadav Brandes',
    author_email  ='nadav.brandes@mail.huji.ac.il',
    packages = ['proteinbert', 'proteinbert.shared_utils'],
    license = 'MIT',
    scripts = [
        'bin/create_uniref_db',
        'bin/create_uniref_h5_dataset',
        'bin/pretrain_proteinbert',
        'bin/set_h5_testset',
    ],
    install_requires = [
        'tensorflow',
        'tensorflow_addons',
        'numpy',
        'pandas',
        'h5py',
        'lxml',
        'pyfaidx',
    ],
)