# Neural-Projection-Skip-Gram with DAN

This repository contains end-to-end implementation of Neural Projection with Skip Gram (NP-SG) and Deep Averaging Network (DAN) model. We first train a NP-SG model and then leverage the trained projected embeddings to train a DAN for SST-fine classification task. Our goal is to compare the performance with embeddings from Language models.

### Steps

1. You need to set the input parameters for in `config.py` 
2. Run `pip install -r requirements.txt `
3. Run `python setup.py`

TO BE UPDATED SOON

### Datasets

We have initially developed the pipeline to use SST-Fine, and Bible Corpus for training the NP-SG model and SST-Fine dataset for classification task.

1. SST-Fine
2. Bible corpus (from nltk)
3. enwiki9 - (TODO)

### References

1. Neural Projection Skip-Gram (https://arxiv.org/pdf/1906.01605.pdf)
2. Deep Averaging Network (https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf)
