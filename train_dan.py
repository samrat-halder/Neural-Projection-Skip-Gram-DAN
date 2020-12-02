from config import *
import numpy as np

from functions.utils import *
from functions.preprocess import *

if __name__ == "__main__":

    pp = PreProcessor(input_train='./data/sst_train.txt', input_val='./data/sst_dev.txt')
    pp.tokenize()
    data_train, labels_train, data_val, labels_val = pp.make_data()

    embedding_matrix = np.load(f'./data/emebddings_{n}_{data}.npy')
    embedding_matrix = embedding_matrix.reshape(embedding_matrix.shape[0], embedding_matrix.shape[1])

    model = model_dan(len(pp.word_index)+1, embedding_matrix, pp.MAX_SEQUENCE_LENGTH, embedding_dim=100, num_class=labels_train.shape[1])
    # get_embedding_layer_output = K.function([model.layers[0].input],[model.layers[0].output])
    # el_output = np.mean(get_embedding_layer_output([data])[0],axis=1)
    # print el_output
    # get_average_word_layer_output = K.function([model.layers[0].input],[model.layers[1].output])
    # print get_average_word_layer_output([data])[0]

    model.fit(data_train,labels_train,batch_size=batch_size,epochs=num_epochs,validation_data=(data_val,labels_val))
