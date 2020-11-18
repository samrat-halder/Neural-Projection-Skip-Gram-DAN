import re
import nltk
import numpy as np
from string import punctuation

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

