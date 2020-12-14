from config import *
from functions.utils import *
from functions.projection import *
from functions.preprocess import *
from functions.models import *

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.random_projection import SparseRandomProjection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter
import scipy.sparse as sp
from operator import itemgetter
import numpy as np

if __name__ == '__main__':

    proc_data = load_pickle( './data', f'proc_data_{n}_{data}.pkl')
    #fit LSH projection pipeline
    print('fitting LSH projections..')

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
    print(len(result))
    print('done!')
    print('_____________________')
    
    print('generating projections for the the vocabulary...')
    pp = PreProcessor(input_train='./data/sst_train.txt', input_val='./data/sst_dev.txt')
    pp.tokenize()
    #data, labels, data_val, labels_val = pp.make_data()
    words_to_project = pp.get_projection_embedding_matrix(PROJECTION_DIM=1120)
    before = pipeline.named_steps["char_term_frequency"].transform(words_to_project)
    after = pipeline.named_steps["union"].transform(before)
    print('saving projection array...')
    np.save(f'./data/projections_sst_fine.npy', after[0].toarray())
    del after
    del before
    del words_to_project
    # print(after[0].toarray().shape)
    # print(words_to_project)
    # print(after[0].toarray()[:10])
    print('done!')
    print('_____________________')
    
    #build the model
    np_sg_model = model_skip_gram(projection_dim=1120, emb_dim=100)
    id2word = load_pickle( './data', f'id2w_{n}_{data}.pkl')
    skip_grams = load_pickle('./data', f'skip_grams_{n}_{data}.pkl')

    print('model training started...')
    #train
    for epoch in range(1, num_epoch+1):
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

            if i % 100 == 0 and i != 0:
                print(f'\tProcessed skip-grams from {i} sentences | Loss: {loss}')
            loss += np_sg_model.train_on_batch(X,Y) 

    del skip_grams
    #prepare embedding matrix for DAN training
    print('preparing embedding matrix...')
    projection_matrix = np.load(f'./data/projections_sst_fine.npy')
    get_embedding_layer_output = K.function([np_sg_model.layers[0].input],
                                                [np_sg_model.layers[3].output])
    embedding_output = get_embedding_layer_output([projection_matrix])[0]
    print(embedding_output.shape)
    np.save(f'./data/emebddings_sst_fine.npy', embedding_output)
