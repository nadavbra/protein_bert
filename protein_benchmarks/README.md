These are the processed benchmark datasets used in the paper. Exact sources described in full paper.

### Citations <a name="citations"></a>
If you find the datasets useful in your research, we ask that you please cite both this work and the relevant source:

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


```bibtex
@inproceedings{Rao2019,
abstract = {Machine learning applied to protein sequences is an increasingly popular area of research. Semi-supervised learning for proteins has emerged as an important paradigm due to the high cost of acquiring supervised protein labels, but the current literature is fragmented when it comes to datasets and standardized evaluation techniques. To facilitate progress in this field, we introduce the Tasks Assessing Protein Embeddings (TAPE), a set of five biologically relevant semi-supervised learning tasks spread across different domains of protein biology. We curate tasks into specific training, validation, and test splits to ensure that each task tests biologically relevant generalization that transfers to real-life scenarios. We benchmark a range of approaches to semi-supervised protein representation learning, which span recent work as well as canonical sequence learning techniques. We find that self-supervised pretraining is helpful for almost all models on all tasks, more than doubling performance in some cases. Despite this increase, in several cases features learned by self-supervised pretraining still lag behind features extracted by state-of-the-art non-neural techniques. This gap in performance suggests a huge opportunity for innovative architecture design and improved modeling paradigms that better capture the signal in biological sequences. TAPE will help the machine learning community focus effort on scientifically relevant problems. Toward this end, all data and code used to run these experiments are available at https://github.com/songlab-cal/tape.},
archivePrefix = {arXiv},
arxivId = {1906.08230},
author = {Rao, Roshan and Bhattacharya, Nicholas and Thomas, Neil and Duan, Yan and Chen, Xi and Canny, John and Abbeel, Pieter and Song, Yun S.},
booktitle = {Advances in Neural Information Processing Systems},
eprint = {1906.08230},
file = {::},
issn = {10495258},
month = {jun},
pmid = {33390682},
title = {{Evaluating protein transfer learning with TAPE}},
url = {https://arxiv.org/abs/1906.08230},
volume = {32},
year = {2019}
}
```

```bibtex
@article{Ofer2015,
abstract = {MOTIVATION: The amount of sequenced genomes and proteins is growing at an unprecedented pace. Unfortunately, manual curation and functional knowledge lag behind. Homologous inference often fails at labeling proteins with diverse functions and broad classes. Thus, identifying high-level protein functionality remains challenging. We hypothesize that a universal feature engineering approach can yield classification of high-level functions and unified properties when combined with machine learning approaches, without requiring external databases or alignment.

RESULTS: In this study, we present a novel bioinformatics toolkit called ProFET (Protein Feature Engineering Toolkit). ProFET extracts hundreds of features covering the elementary biophysical and sequence derived attributes. Most features capture statistically informative patterns. In addition, different representations of sequences and the amino acids alphabet provide a compact, compressed set of features. The results from ProFET were incorporated in data analysis pipelines, implemented in python and adapted for multi-genome scale analysis. ProFET was applied on 17 established and novel protein benchmark datasets involving classification for a variety of binary and multi-class tasks. The results show state of the art performance. The extracted features' show excellent biological interpretability. The success of ProFET applies to a wide range of high-level functions such as subcellular localization, structural classes and proteins with unique functional properties (e.g. neuropeptide precursors, thermophilic and nucleic acid binding). ProFET allows easy, universal discovery of new target proteins, as well as understanding the features underlying different high-level protein functions.

AVAILABILITY AND IMPLEMENTATION: ProFET source code and the datasets used are freely available at https://github.com/ddofer/ProFET.

CONTACT: michall@cc.huji.ac.ilSupplementary information: Supplementary data are available at Bioinformatics online.},
author = {Ofer, Dan and Linial, Michal},
doi = {10.1093/bioinformatics/btv345},
issn = {1367-4811},
journal = {Bioinformatics (Oxford, England)},
keywords = {NeuroPID,ProFET,Proteins},
mendeley-tags = {NeuroPID,ProFET,Proteins},
month = {jun},
pmid = {26130574},
title = {{ProFET: Feature engineering captures high-level protein functions.}},
url = {http://bioinformatics.oxfordjournals.org/content/early/2015/07/02/bioinformatics.btv345.abstract},
year = {2015}
}
```

```bibtex
@article{OferD2014,
abstract = {MOTIVATION: The evolution of multicellular organisms is associated with increasing variability of molecules governing behavioral and physiological states. This is often achieved by neuropeptides (NPs) that are produced in neurons from a longer protein, named neuropeptide precursor (NPP). The maturation of NPs occurs through a sequence of proteolytic cleavages. The difficulty in identifying NPPs is a consequence of their diversity and the lack of applicable sequence similarity among the short functionally related NPs. RESULTS: Herein, we describe Neuropeptide Precursor Identifier (NeuroPID), a machine learning scheme that predicts metazoan NPPs. NeuroPID was trained on hundreds of identified NPPs from the UniProtKB database. Some 600 features were extracted from the primary sequences and processed using support vector machines (SVM) and ensemble decision tree classifiers. These features combined biophysical, chemical and informational-statistical properties of NPs and NPPs. Other features were guided by the defining characteristics of the dibasic cleavage sites motif. NeuroPID reached 89-94{\%} accuracy and 90-93{\%} precision in cross-validation blind tests against known NPPs (with an emphasis on Chordata and Arthropoda). NeuroPID also identified NPP-like proteins from extensively studied model organisms as well as from poorly annotated proteomes. We then focused on the most significant sets of features that contribute to the success of the classifiers. We propose that NPPs are attractive targets for investigating and modulating behavior, metabolism and homeostasis and that a rich repertoire of NPs remains to be identified. AVAILABILITY: NeuroPID source code is freely available at http://www.protonet.cs.huji.ac.il/neuropid},
author = {{Ofer D} and {Linial M} and Ofer, Dan and Linial, Michal},
doi = {10.1093/bioinformatics/btt725},
issn = {1367-4811},
journal = {Bioinformatics (Oxford, England)},
month = {mar},
number = {7},
pages = {931--40},
pmid = {24336809},
title = {{NeuroPID: a predictor for identifying neuropeptide precursors from metazoan proteomes.}},
url = {http://www.ncbi.nlm.nih.gov/pubmed/24336809 https://www.pubchase.com/article/24336809},
volume = {30},
year = {2014}
}
```

```bibtex
@article{Karsenty2014,
abstract = {Neuropeptides (NPs) are short secreted peptides produced in neurons. NPs act by activating signaling cascades governing broad functions such as metabolism, sensation and behavior throughout the animal kingdom. NPs are the products of multistep processing of longer proteins, the NP precursors (NPPs). We present NeuroPID (Neuropeptide Precursor Identifier), an online machine-learning tool that identifies metazoan NPPs. NeuroPID was trained on 1418 NPPs annotated as such by UniProtKB. A large number of sequence-based features were extracted for each sequence with the goal of capturing the biophysical and informational-statistical properties that distinguish NPPs from other proteins. Training several machine-learning models, including support vector machines and ensemble decision trees, led to high accuracy (89-94{\%}) and precision (90-93{\%}) in cross-validation tests. For inputs of thousands of unseen sequences, the tool provides a ranked list of high quality predictions based on the results of four machine-learning classifiers. The output reveals many uncharacterized NPPs and secreted cell modulators that are rich in potential cleavage sites. NeuroPID is a discovery and a prediction tool that can be used to identify NPPs from unannotated transcriptomes and mass spectrometry experiments. NeuroPID predicted sequences are attractive targets for investigating behavior, physiology and cell modulation. The NeuroPID web tool is available at http:// neuropid.cs.huji.ac.il.},
author = {Karsenty, S. and Rappoport, N. and Ofer, D. and Zair, A. and Linial, M.},
doi = {10.1093/nar/gku363},
issn = {0305-1048},
journal = {Nucleic Acids Research},
keywords = {NPID web tool NeuroPID NAR NP NPP neuropeptides},
mendeley-tags = {NPID web tool NeuroPID NAR NP NPP neuropeptides},
month = {may},
pages = {gku363--},
title = {{NeuroPID: a classifier of neuropeptide precursors}},
url = {http://nar.oxfordjournals.org/content/early/2014/05/03/nar.gku363.abstract},
year = {2014}
}
```

```bibtex
@article{Brandes2016,
abstract = {Determining residue-level protein properties, such as sites of post-translational modifications (PTMs), is vital to understanding protein function. Experimental methods are costly and time-consuming, while traditional rule-based computational methods fail to annotate sites lacking substantial similarity. Machine Learning (ML) methods are becoming fundamental in annotating unknown proteins and their heterogeneous properties. We present ASAP (Amino-acid Sequence Annotation Prediction), a universal ML framework for predicting residue-level properties. ASAP extracts numerous features from raw sequences, and supports easy integration of external features such as secondary structure, solvent accessibility, intrinsically disorder or PSSM profiles. Features are then used to train ML classifiers. ASAP can create new classifiers within minutes for a variety of tasks, including PTM prediction (e.g. cleavage sites by convertase, phosphoserine modification). We present a detailed case study for ASAP: CleavePred, an ASAP-based model to predict protein precursor cleavage sites, with state-of-the-art results. Protein cleavage is a PTM shared by a wide variety of proteins sharing minimal sequence similarity. Current rule-based methods suffer from high false positive rates, making them suboptimal. The high performance of CleavePred makes it suitable for analyzing new proteomes at a genomic scale. The tool is attractive to protein design, mass spectrometry search engines and the discovery of new bioactive peptides from precursors. ASAP functions as a baseline approach for residue-level protein sequence prediction. CleavePred is freely accessible as a web-based application. Both ASAP and CleavePred are open-source with a flexible Python API.},
author = {Brandes, Nadav and Ofer, Dan and Linial, Michal},
doi = {10.1093/database/baw133},
issn = {17580463},
journal = {Database},
title = {{ASAP: A machine learning framework for local protein properties}},
volume = {2016},
year = {2016}
}

```


