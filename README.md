# Neural-Projection Skip-Gram with Deep Average Network Classifier

This repository contains end-to-end implementation of Neural Projection with Skip Gram (NP-SG) and Deep Averaging Network (DAN) model. We first train a NP-SG model and then leverage the trained projected embeddings to train a DAN for SST-fine classification task. Our goal is to compare the performance of projection based on the fly embeddings generated from locally sensitive hashing with static and non-static embeddings.

### Steps


1. Clone the repository.

2. Run `pip install -r requirements.txt `. This will download and install all the required packages.

3. Run `python setup.py`. This will set-up the directory structure and download required corpora for experiments.

4. Request for enWiki9 dataset at sh3970@columbia.edu. 

4. Set the `config.py` script before running experiments. The experiments spans over two steps - \
  a. Training a NP-SG model with some corpus (we use a chunk of wiki9). The larger this corpus the better it is.\
  b. Using the embeddings from step 1, for any downstream task e.g. we train a DAN model for SST-fine data
  
5. To test the setup set n=1000, test=True. 

6. To run a complete experiement run the following three scripts:\
  a. `python3 data_prep.py`\
  b. `python3 train_projection.py`\
  c. `python3 train_dan.py` \
  OR \
  you can run the bash script `run.sh` 
  
7. Set n > 10,000 to train the NP-SG model on a larger corpus.



## Results

| Trainable Embedding? | NP-SG train Dataset | Skip-gram train Size | Test Acc. (SST-Fine) |
| :---: | :---: | :---: | :---: | 
| No | SST-Fine | 7,000 | 30.9% |
| Yes | SST-Fine | 7,000 | 37.68% |
| No | enWiki9 | 1,000 | 27.88% | 
| Yes | enWiki9 | 1,000 | 37.51% | 
| No | enWiki9 | 5,000 | 29.7% |
| Yes | enWiki9 | 5,000 | 38.1% |
| No | enWiki9 | 30,000 | 30.43% |
| Yes | enWiki9 | 30,000 | 38.42% | 
| No | enWiki9 | 60,000 | 30.97% |
| Yes | enWiki9 | 60,000 | 40.33% | 


### Datasets

We have initially developed the pipeline to use wiki9, SST-Fine, and Bible Corpus for training the NP-SG model alongwith a DAN model on SST-Fine dataset for five class classification task.

1. SST-Fine (to train test classification task)
2. Bible corpus (from nltk)
3. enwiki9 (to train NP-SG model)

### References

1. Neural Projection Skip-Gram (https://arxiv.org/pdf/1906.01605.pdf)
2. Deep Averaging Network (https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf)
