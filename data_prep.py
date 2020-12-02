from config import *
from nltk.corpus import gutenberg
from functions.preprocess import *
from functions.projection import *
from functions.utils import *
import pytreebank
import os
import pandas as pd
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams

if __name__ == '__main__':

    if data == 'bible_corpus':
        raw_data = gutenberg.sents('bible-kjv.txt') 
    elif data == 'sst_fine':
        # Print the length of the training set
        #print(len(dataset['train']))
        df = pd.read_csv('./data/sst_train.txt', sep='\t', header=None, names=['truth', 'text'])
        df['truth'] = df['truth'].str.replace('__label__', '')
        df['truth'] = df['truth'].astype(int).astype('category')
        print(df[:2])
        raw_data = [sent.split(' ') for sent in list(df['text'])]

    proc_data = preprocess(raw_data)
    if test:
        print(f'total size of train data for {data} corpus is {len(proc_data)} but this is test and we take first {n} examples')
        proc_data = proc_data[:n]
    print('processing data to generate skip-gram pairs..') 
    print(proc_data[:2])
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(proc_data)
    word2id = tokenizer.word_index
    id2word = {v:k for k, v in word2id.items()}
    vocab_size = len(word2id) + 1
    wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in proc_data]
    print('Vocabulary Size:', vocab_size)
    print('Vocabulary Sample:', list(word2id.items())[:10])

    # generate skip-grams
    skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=10) for wid in wids]

    # view sample skip-grams
    pairs, labels = skip_grams[0][0], skip_grams[0][1]
    for i in range(10):
        print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
                  id2word[pairs[i][0]], pairs[i][0], 
                  id2word[pairs[i][1]], pairs[i][1], 
                  labels[i]))
    print('saving..')

    dump_pickle(proc_data, './data', f'proc_data_{n}_{data}.pkl')
    dump_pickle(skip_grams, './data', f'skip_grams_{n}_{data}.pkl')
    dump_pickle(word2id, './data', f'w2id_{n}_{data}.pkl')
    dump_pickle(id2word, './data', f'id2w_{n}_{data}.pkl')

    print('all finished!')
