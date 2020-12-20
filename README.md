# Neural-Projection Skip-Gram with Deep Average Network Classifier

This repository contains end-to-end implementation of Neural Projection with Skip Gram (NP-SG) and Deep Averaging Network (DAN) model. We first train a NP-SG model and then leverage the trained projected embeddings to train a DAN for SST-fine classification task. Our goal is to compare the performance of projection based on the fly embeddings generated from locally sensitive hashing with static and non-static embeddings.

### Steps


1. Clone the repository.

2. Run `pip install -r requirements.txt `. This will download and install all the required packages.

3. Run `python setup.py`. This will set-up the directory structure and download required corpora for experiments.

4. Download wiki9 data from https://drive.google.com/file/d/1IxrDntl73wrQx3yxzN6rIIsRMFQDC-9e/view?usp=sharing (hosted on my Google Drive) to the data directory and untar it. 

4. Set the `config.py` script before running experiments. The experiments spans over two steps - \
  a. Training a NP-SG model with some corpus (we use a chunk of wiki9). The larger this corpus the better it is.\
  b. Using the embeddings from step 1, for any downstream task e.g. we train a DAN model for SST-fine data
  
5. To test the setup set n=1000, test=True. 

6. To run a complete experiement run the following three scripts:\
  a. `python3 data_prep.py`\
  b. `python3 train_projection.py`\
  c. `python3 train_dan.py`
  
7. Set n >= 10,000 to train the NP-SG model on alarger dataset.



**Results**

| Trainable Embedding? | NP-SG train Dataset | Size | Test Acc. (SST-Fine) |
| :---: | :---: | :---: | :---: | :---: |
| No | SST-Fine | 7000 | 30.9% |
| Yes | SST-Fine | 7000 | 37.68% |
| No | Wiki-9 | 1000 | 27.88% | 
| Yes | Wiki-9 | 1000 | 38.51% | 
| No | Wiki-9 | 5000 | 29.7% |
| Yes | Wiki-90| 5000 | 39.1% |


| Class | Precision	| Recall	| f1 |
| :---: | :---: | :---: | :---: |
| INTP | 0.47	| 0.45 | 0.46 |
| INTJ | 0.39 |	0.69 | 	0.5 |
| INFP | 0.35 |	0.27 |	0.31 |
| INFJ | 0.4 | 0.24 |	0.3 |
| ENTP | 0.55 |	0.12 | 0.2 |
| ENFP | 0.37 |	0.1 |	0.15 |
| ENTJ | 0.38 |	0.05 | 0.09 |
| ENFJ | 0.28 |	0.04 | 0.07 |

### Datasets

We have initially developed the pipeline to use wiki9, SST-Fine, and Bible Corpus for training the NP-SG model alongwith a DAN model on SST-Fine dataset for five class classification task.

1. SST-Fine (to train test classification task)
2. Bible corpus (from nltk)
3. enwiki9 (to train NP-SG model)

### References

1. Neural Projection Skip-Gram (https://arxiv.org/pdf/1906.01605.pdf)
2. Deep Averaging Network (https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf)
