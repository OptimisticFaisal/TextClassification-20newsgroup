from keras.layers import Embedding,Dropout,Conv1D,MaxPooling1D,LSTM,Dense,CuDNNLSTM
#from keras.models import Sequential
from utility_class import Utility
from keras.utils import np_utils
from keras.initializers import Constant
from keras.utils import plot_model
from keras.models import Sequential
import numpy as np

categories = ['com', 'rec', 'religion', 'politics']
util = Utility()
X, Y =util.get_data(categories)
docs = util.preprocessing_dataset(X)
del X
padded_doc,Y,word_index, max_length = util.pad_input(docs,Y)
#word_embedding_file_name = 'word2vector2.txt'
#avg_emb,emb_mat = util.average_embedding(word_index,word_embedding_file_name)
file_name = 'glove.6B.300d.txt'
embedding_matrix = util.create_embedding_matrix(file_name,word_index,300)
vocabulary_size = len(word_index)+1


def create_conv_model():
    model_conv = Sequential()
    embedding_layer = Embedding(vocabulary_size, 300, embeddings_initializer=Constant(embedding_matrix), input_length=max_length,
                                trainable=False)
    model_conv.add(embedding_layer)
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 2,activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(50))
    model_conv.add(Dense(len(categories), activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy', optimizer='adam',    metrics=['accuracy',util.precision_m,util.recall_m,util.f1_m])
    #plot_model(model_conv, to_file='conv_lstm_model.png', show_shapes=True)
    model_conv.summary()
    return model_conv


mod = create_conv_model()
train_x =padded_doc[:10000]
test_x = padded_doc[10001:]
train_y = np.asarray(Y[:10000])
encoded_train_y = np_utils.to_categorical(train_y)
test_y = np.asarray(Y[10001:])
encoded_test_y = np_utils.to_categorical(test_y)

mod.fit(train_x,encoded_train_y,epochs=20, batch_size=10)
accuracy =mod.evaluate(test_x,encoded_test_y)
print(accuracy)