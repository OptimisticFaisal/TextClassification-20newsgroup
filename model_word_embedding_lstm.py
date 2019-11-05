from utility_class import Utility
from keras.utils import np_utils
from keras.initializers import Constant
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Embedding,Dense,LSTM,CuDNNLSTM
import tensorflow as tf
import numpy as np
categories = ['com', 'rec', 'religion', 'politics']
util = Utility()
X, Y =util.get_data(categories)
docs = util.preprocessing_dataset(X)
del X
padded_doc,Y,word_index, max_length = util.pad_input(docs, Y)
#word_embedding_file_name = 'word2vector2.txt'
#avg_emb,emb_mat = util.average_embedding(word_index,word_embedding_file_name)
file_name = 'glove.6B.300d.txt'
embedding_matrix = util.create_embedding_matrix(file_name,word_index,300)
num_words = len(word_index)+1


def base_model():
    model = Sequential()
    embedding_layer = Embedding(num_words,300, embeddings_initializer=Constant(embedding_matrix), input_length=max_length,trainable=False)
    model.add(embedding_layer)
    #model.add(Embedding(len(word_index),128,input_length=max_length))
    model.add(CuDNNLSTM(100))
    model.add(Dense(len(categories), activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',util.precision_m,util.recall_m,util.f1_m])
    #tf.keras.utils.plot_model(model, to_file='lstm_model.png', show_shapes=True)

    model.summary()
    return model

mod =base_model()

train_x =padded_doc[:10000]
test_x = padded_doc[10001:]
train_y = np.asarray(Y[:10000])
encoded_train_y = np_utils.to_categorical(train_y)
test_y = np.asarray(Y[10001:])
encoded_test_y = np_utils.to_categorical(test_y)

mod.fit(train_x,encoded_train_y,validation_data=(test_x,encoded_test_y),epochs=20, batch_size=10,)
accuracy =mod.evaluate(test_x,encoded_test_y)
print(accuracy)