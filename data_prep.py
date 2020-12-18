from config import *
from nltk.corpus import gutenberg
from functions.preprocess import *
from functions.projection import *
from functions.utils import *
import pytreebank
import os
import sys
import glob
import pandas as pd
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams

if __name__ == '__main__':
    if data == 'bible_corpus':
        raw_data = gutenberg.sents('bible-kjv.txt') 
        proc_data = preprocess(raw_data)
    elif data == 'sst_fine':
        df = pd.read_csv('./data/sst_train.txt', sep='\t', header=None, names=['truth', 'text'])
        df['truth'] = df['truth'].str.replace('__label__', '')
        df['truth'] = df['truth'].astype(int).astype('category')
        print(df[:2])
        raw_data = [sent.split(' ') for sent in list(df['text'])]
        proc_data = preprocess(raw_data)
    elif data == 'wiki9':
        raw_data = []
        for file in glob.glob('./data/wiki9/articles/*')[:n]:
            with open(file, 'r') as f:
                raw_data += f.readlines()
            f.close()
        raw_data = [sent.split(' ') for sent in raw_data]
        proc_data = [' '.join(l) for l in raw_data]
        # print(raw_data[:2])
    # proc_data = preprocess(raw_data)
    del raw_data
    if test:
        print('-------------------------')
        print(f'total size of train data for {data} corpus is {len(proc_data)} but this is test and we take first {n} examples\n')
        proc_data = proc_data[:n]
    else:
        n = len(proc_data)
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
    # print(wids)
    # generate skip-grams
    print('saving processed data and cleaning memory...')
    dump_pickle(proc_data, './data', f'proc_data_{n}_{data}.pkl')
    dump_pickle(word2id, './data', f'w2id_{n}_{data}.pkl')
    dump_pickle(id2word, './data', f'id2w_{n}_{data}.pkl')
    del proc_data
    del word2id

    print('generating skip-grams...')
    skip_gram_path = f'./data/skip_grams_{n}_{data}'

    if not os.path.exists(skip_gram_path):
        os.makedirs(skip_gram_path)

    skip_grams = []
    sent_i = 0
    batch = 1
    for wid in wids:
        skip_grams.append(skipgrams(wid, vocabulary_size=vocab_size, window_size=window_size))
        # print(skip_grams)

        sent_i += 1
        sys.stdout.write('.')
        if sent_i == 100:
            # view sample skip-grams
            pairs, labels = skip_grams[0][0], skip_grams[0][1]
            print('\nsample skip-grams:')
            for i in range(3):
                print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
                    id2word[pairs[i][0]], pairs[i][0], 
                    id2word[pairs[i][1]], pairs[i][1], 
                    labels[i]))
            print(f'saving batch {batch}. Completed {round(sent_i*batch*100/len(wids),2)}%')
            dump_pickle(skip_grams, skip_gram_path, f'batch_id_{batch}.pkl')
            batch += 1
            sent_i = 0
            skip_grams = []
    
    print('all finished!')
