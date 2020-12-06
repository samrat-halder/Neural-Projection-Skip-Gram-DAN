# Neural-Projection-Skip-Gram with DAN

This repository contains end-to-end implementation of Neural Projection with Skip Gram (NP-SG) and Deep Averaging Network (DAN) model. We first train a NP-SG model and then leverage the trained projected embeddings to train a DAN for SST-fine classification task. Our goal is to compare the performance of projection based on the fly embeddings generated from locally sensitive hashing with static and non-static embeddings.

### Steps

1. You need to set the input parameters for in `config.py` 
2. Run `pip install -r requirements.txt `. This will download and install all the required packages.
3. Run `python setup.py`. This will set-up the directory structure and download required corpora for experiments.
4. Now the user needs to set the `config.py` script before running experiments. The experiments spans over two steps - 
  a. Training a NP-SG model with some corpus (we use a chunk from the SST-fine training data set). The larger this corpus the better it is.
  b. Using the embeddings from step 1, for any downstream task e.g. we train a DAN model for SST-fine data
  
  A typical `config` file looks like this:
  `test = True
  n = 5000 #first n sentence if test
  data = 'sst_fine' #'bible_corpus' #sst_fine
  #NP-SG setting
  window_size=5
  num_epoch = 5

  char_ngram_range = (1, 4)
  char_term_frequency_params = {
        'char_term_frequency__analyzer': 'char',
        'char_term_frequency__lowercase': True,
        'char_term_frequency__ngram_range': char_ngram_range,
        'char_term_frequency__strip_accents': None,
        'char_term_frequency__min_df': 2,
        'char_term_frequency__max_df': 0.99,
        'char_term_frequency__max_features': int(1e7),
    }
  T = 80
  d = 14

  embedding_dim = 100
  num_hidden_layers = 3
  num_hidden_units = 500
  num_epochs = 100
  batch_size = 128
  dropout_rate = 0.2
  word_dropout_rate = 0.3
  activation = 'relu'`

TO BE UPDATED SOON

### Datasets

We have initially developed the pipeline to use SST-Fine, and Bible Corpus for training the NP-SG model alongwith a DAN model on SST-Fine dataset for classification task.

1. SST-Fine
2. Bible corpus (from nltk)
3. enwiki9 - (TODO)

### References

1. Neural Projection Skip-Gram (https://arxiv.org/pdf/1906.01605.pdf)
2. Deep Averaging Network (https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf)
