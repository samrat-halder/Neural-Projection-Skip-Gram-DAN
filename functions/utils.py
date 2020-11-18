import pickle
import os
#tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import dot

def dump_pickle(obj, data_dir, fname):
    with open(os.path.join(data_dir, fname), 'wb') as f:
        pickle.dump(obj, f)
    f.close()

def load_pickle(data_dir, fname):
    with open(os.path.join(data_dir, fname), 'rb') as f:
        obj = pickle.load(f)
    return obj

def model(projection_dim=1120, emb_dim=100):
    input_target = Input((projection_dim,))
    input_context = Input((projection_dim,))
    model = tf.keras.models.Sequential()
    # add input layer
    model.add(tf.keras.layers.Dense(
        units=2048,
        input_dim=projection_dim,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='relu') 
    )
    # add hidden layer
    model.add(
        tf.keras.layers.Dense(
        units=100,
        input_dim=2048,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='relu')
    )
    word_embedding = model(input_target)
    word_embedding = Reshape((emb_dim, 1))(word_embedding)
    context_embedding = model(input_context)
    context_embedding = Reshape((emb_dim, 1))(context_embedding)
    # perform the dot product operation  
    dot_product = dot([word_embedding, context_embedding], axes=1, normalize=True)
    dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)
    model1 = Model([input_target, input_context], output)
    model1.compile(loss='mean_squared_error', optimizer='rmsprop')
    # view model summary
    print(model1.summary())

    return model1
