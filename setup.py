import nltk
import os
nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('punkt')

data_path = './data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

model_path = './model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
