from functions.utils import *
from config import *
from functions.projection import *
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.random_projection import SparseRandomProjection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from pprint import pprint
import scipy.sparse as sp
from operator import itemgetter
import numpy as np

if __name__ == '__main__':

    proc_data = load_pickle( './data', f'proc_data_{n}_{data}.pkl')
    #fit LSH projection pipeline
    print('fitting LSH projections..')
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

    hashing_feature_union_params = {
        # T=80 projections for each of dimension d=14: 80 * 14 = 1120-dimensionnal word projections.
        **{'union__sparse_random_projection_hasher_{}__n_components'.format(t): d
           for t in range(T)
        },
        **{'union__sparse_random_projection_hasher_{}__dense_output'.format(t): False  # only AFTER hashing.
           for t in range(T)
        }
    }
    params = dict()
    params.update(char_term_frequency_params)
    params.update(hashing_feature_union_params)
    pipeline = Pipeline([
        ("word_tokenizer", WordTokenizer()),
        ("char_term_frequency", CountVectorizer3D()),
        ('union', FeatureUnion3D([
            ('sparse_random_projection_hasher_{}'.format(t), SparseRandomProjection())
            for t in range(T)
        ]))
    ])
    pipeline.set_params(**params)
    result = pipeline.fit_transform(proc_data)
    print('done!')
    print(len(result))
    #build the model
    np_sg_model = model(projection_dim=1120, emb_dim=100)
    id2word = load_pickle( './data', f'id2w_{n}_{data}.pkl')
    skip_grams = load_pickle('./data', f'skip_grams_{n}_{data}.pkl')

    print('model training started...')
    #train
    for epoch in range(1, num_epoch):
        print('Epoch:', epoch)
        loss = 0
        for i, elem in enumerate(skip_grams):
            pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
            pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
            first_words_to_hash = [list(itemgetter(*pair_first_elem)(id2word))]
            before_1 = pipeline.named_steps["char_term_frequency"].transform(first_words_to_hash)
            after_1 = pipeline.named_steps["union"].transform(before_1)

            second_words_to_hash = [list(itemgetter(*pair_first_elem)(id2word))]
            before_2 = pipeline.named_steps["char_term_frequency"].transform(second_words_to_hash)
            after_2 = pipeline.named_steps["union"].transform(before_2)

            labels = np.array(elem[1], dtype='int32')
            X = [after_1[0].toarray(), after_2[0].toarray()]
            Y = labels

            if i % 1000 == 0:
                print('\tProcessed {} skip-grams'.format(i))
            loss += np_sg_model.train_on_batch(X,Y)  

        print('Loss:', round(loss,4))
