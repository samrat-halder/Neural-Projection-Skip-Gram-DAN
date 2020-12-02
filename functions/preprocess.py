import re
import nltk
import numpy as np
import pandas as pd
import pickle as pkl
from string import punctuation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

def preprocess(doc):
    remove_terms = punctuation + '0123456789'

    norm_doc = [[word.lower() for word in sent if word not in remove_terms] for sent in doc]
    norm_doc = [' '.join(tok_sent) for tok_sent in norm_doc]
    norm_doc = filter(None, normalize_corpus(norm_doc))
    norm_doc = [tok_sent for tok_sent in norm_doc if len(tok_sent.split()) > 2]

    print('Total lines:', len(doc))
    print('\nSample line:', doc[0])
    print('\nProcessed line:', norm_doc[0])

    return norm_doc

class PreProcessor(object):
    def __init__(self,input_train='../data/sst_train.txt', input_val='../data/sst_val.txt'):
        self.input_train = input_train
        self.input_val = input_val

    def tokenize(self):

        train_df = pd.read_csv(self.input_train, sep='\t', header=None, names=['truth', 'text'])
        train_df['truth'] = train_df['truth'].str.replace('__label__', '')
        #train_df['truth'] = train_df['truth'].astype(int).astype('category')
        
        val_df = pd.read_csv(self.input_val, sep='\t', header=None, names=['truth', 'text'])
        val_df['truth'] = val_df['truth'].str.replace('__label__', '')
        #val_df['truth'] = val_df['truth'].astype(int).astype('category')

        self.train_data = train_df.to_numpy()
        self.val_data = val_df.to_numpy()

        sent_train = np.array(["".join(self.train_data[i,1]) for i in range(self.train_data.shape[0])])
        sent_val = np.array(["".join(self.val_data[i,1]) for i in range(self.val_data.shape[0])])

        print(sent_train[0])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sent_train)

        self.sequences = tokenizer.texts_to_sequences(sent_train)
        self.sequences_val = tokenizer.texts_to_sequences(sent_val)

        self.word_index = tokenizer.word_index
        print(f"Found {len(self.word_index)} unique tokens")

    def make_data(self):
        self.MAX_SEQUENCE_LENGTH = max([len(self.sequences[i]) for i in range(len(self.sequences))])

        data = pad_sequences(self.sequences,maxlen=self.MAX_SEQUENCE_LENGTH)
        data_val = pad_sequences(self.sequences_val,maxlen=self.MAX_SEQUENCE_LENGTH)

        classes_train = set(self.train_data[:,0])
        classes_val = set(self.val_data[:,0])
        classes = classes_train.union(classes_val)

        labels_index = {} 
        classes_index = {} 

        for i,j in enumerate(classes):
            labels_index[j] = i
            classes_index[i] = j

        labels = np.zeros((len(self.sequences),1))
        labels_val = np.zeros((len(self.sequences_val),1))

        for i in range(len(self.sequences)):
            labels[i] = labels_index[self.train_data[i,0]]

        for i in range(len(self.sequences_val)):
            labels_val[i] = labels_index[self.val_data[i,0]]

        labels = to_categorical(labels,num_classes=len(classes))
        labels_val = to_categorical(labels_val,num_classes=len(classes))

        print(f"Shape of data tensor: {data.shape}")
        print(f"Shape of label tensor: {labels.shape}")

        return data, labels, data_val, labels_val

    def get_projection_embedding_matrix(self, PROJECTION_DIM=1120):

        projection_wordlist =['']*(len(self.word_index)+1)
        for word, i in self.word_index.items():
            projection_wordlist[i] = word
        
        return [projection_wordlist] 
