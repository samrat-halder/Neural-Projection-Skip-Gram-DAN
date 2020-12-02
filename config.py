test = True
n = 1000 #first n sentence if test
data = 'sst_fine' #'bible_corpus' #sst_fine
num_epoch = 1
# Projection settings
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
# DAN network settings
embedding_dim = 100
num_hidden_layers = 3
num_hidden_units = 500
num_epochs = 100
batch_size = 128
dropout_rate = 0.2
word_dropout_rate = 0.3
activation = 'relu'
