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

if data == 'bible_corpus':
    raw_data = gutenberg.sents('bible-kjv.txt') 
elif data == 'sst_fine':
    out_path = os.path.join('./data', 'sst_{}.txt')
    dataset = pytreebank.load_sst('./raw_data')

    # Store train, dev and test in separate files
    for category in ['train', 'test', 'dev']:
        with open(out_path.format(category), 'w') as outfile:
            for item in dataset[category]:
                outfile.write("__label__{}\t{}\n".format(
                    item.to_labeled_lines()[0][0] + 1,
                    item.to_labeled_lines()[0][1]
                ))
    # Print the length of the training set
    #print(len(dataset['train']))
    df = pd.read_csv('./data/sst_train.txt', sep='\t', header=None, names=['truth', 'text'])
    df['truth'] = df['truth'].str.replace('__label__', '')
    df['truth'] = df['truth'].astype(int).astype('category')
    print(df[:2])
    raw_data = [sent.split(' ') for sent in list(df['text'])]

proc_data = preprocess(raw_data)
if test:
    proc_data = proc_data[:n]

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

dump_pickle(proc_data, './data', f'proc_data_{n}_{data}.pkl')
dump_pickle(skip_grams, './data', f'skip_grams_{n}_{data}.pkl')
dump_pickle(word2id, './data', f'w2id_{n}_{data}.pkl')
dump_pickle(id2word, './data', f'id2w_{n}_{data}.pkl')
