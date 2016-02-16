# embeddings
This repository contains code accompanying publication of the paper: 
> Y. Choi, Y. Chiu, D. Sontag. [Learning Low-Dimensional Representations of Medical Concepts](http://cs.nyu.edu/~dsontag/papers/ChoiChiuSontag_AMIA_CRI16.pdf). To appear in Proceedings of the AMIA Summit on Clinical Research Informatics (CRI), 2016.

In the base directory there are two files containing the two best embeddings learned in the paper:
* `claims_codes_hs_300.txt.gz`: Embeddings of ICD-9 diagnosis and procedure codes, NDC medication codes, and LOINC laboratory codes, learned on a large claims dataset from 2005 to 2013 for roughly 4 million people.
* `claims_cuis_hs_300.txt.gz`: Embeddings of [UMLS](https://www.nlm.nih.gov/research/umls/) concept unique identifiers (CUIs), learned from 20 million clinical notes spanning 19 years of data from Stanford Hospital and Clinics, using a  [data set](http://datadryad.org/resource/doi:10.5061/dryad.jp917) released in a [paper](http://www.nature.com/articles/sdata201432) by Finlayson, LePendu & Shah.
