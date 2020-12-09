import nltk
import pytreebank
import os

nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('punkt')

print('creating directories..')
data_path = './data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

model_path = './model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

print('downloading and formating SST-fine dataset..')
# SST-fine corpus
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
print('all set-up completeed successfully!')

# This will throw an error at the end of the script : TODO: FIXME
from functions.wiki9 import write_wiki9_articles
write_wiki9_articles()
