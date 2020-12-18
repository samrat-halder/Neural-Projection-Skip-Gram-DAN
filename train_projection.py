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
import glob
import pickle
import os

import tensorflow as tf
from keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    # skip_grams = load_pickle('./data', f'skip_grams_{n}_{data}.pkl')

    print('model training started...')
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    #train
    all_batch = [f for f in glob.glob(f'./data/skip_grams_{n}_{data}/*.pkl')]
    for epoch in range(1, num_epoch+1):
        print('Epoch:', epoch)
        loss = 0
        batch_id = 1
        for skip_gram_batch in all_batch:
            with open(skip_gram_batch, 'rb') as f:
                skip_grams = pickle.load(f)
            f.close()

            for i, elem in enumerate(skip_grams):
                # print(i)
                pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
                pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
                if np.size(pair_first_elem) > max_batch_size:
                    print(f'Warning: {np.size(pair_first_elem)} > specified limit of {max_batch_size}. sequence: {batch_id},{i} skipped.')
                    continue
                first_words_to_hash = [list(itemgetter(*pair_first_elem)(id2word))]
                before_1 = pipeline.named_steps["char_term_frequency"].transform(first_words_to_hash)
                after_1 = pipeline.named_steps["union"].transform(before_1)

                second_words_to_hash = [list(itemgetter(*pair_first_elem)(id2word))]
                before_2 = pipeline.named_steps["char_term_frequency"].transform(second_words_to_hash)
                after_2 = pipeline.named_steps["union"].transform(before_2)

                labels = np.array(elem[1], dtype='int32')
                X = [after_1[0].toarray(), after_2[0].toarray()]
                Y = labels
                # print(len(Y))
                # if len(Y) > max_batch_size:
                #    print(f'Warning: number of skip-grams in this batch {i} is {len(Y)} > {max_batch_size}. To avoid OOM skipping this batch')
                #    continue

                # if i % 100 == 0 and i != 0:
                loss += np_sg_model.train_on_batch(X,Y)

            print(f'Loss: {loss} after processing bacth_id {batch_id}')
            batch_id += 1

    tf.compat.v1.keras.backend.get_session().close()

    #prepare embedding matrix for DAN training
    print('preparing embedding matrix...')
    projection_matrix = np.load(f'./data/projections_sst_fine.npy')
    get_embedding_layer_output = K.function([np_sg_model.layers[0].input],
                                                [np_sg_model.layers[3].output])
    embedding_output = get_embedding_layer_output([projection_matrix])[0]
    print(embedding_output.shape)
    np.save(f'./data/emebddings_sst_fine.npy', embedding_output)
