from utility_class import Utility
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Embedding,LSTM, Conv1D,Flatten,GlobalMaxPool1D, Concatenate, InputLayer
from keras.layers import concatenate, Activation, Dense,Input,Add
from keras.layers import cudnn_recurrent
from keras.layers import CuDNNLSTM
from keras.initializers import Constant
from keras.utils import plot_model
import numpy as np

categories = ['com', 'rec', 'religion', 'politics']
util = Utility()
X, Y =util.get_data(categories)
docs = util.preprocessing_dataset(X)
del X
padded_doc,Y, word_index, max_length = util.pad_input(docs,Y)
#word_embedding_file_name = 'word2vector2.txt'
#avg_emb,emb_mat = util.average_embedding(word_index,word_embedding_file_name)
file_name = 'glove.6B.300d.txt'
embedding_matrix = util.create_embedding_matrix(file_name,word_index,300)
num_words = len(word_index)+1

def base_model():
    model = Sequential()
    embedding_layer = Embedding(num_words,50, embeddings_initializer=Constant(embedding_matrix), input_length=max_length,trainable=False)
    model.add(embedding_layer)

    #model.add(Embedding(len(word_index),128,input_length=max_length))
    #model.add(Conv2D(1,kernel_size=(2,300),activation='relu',data_format="channels_first",input_shape=(10,1,max_length,300)))
    model.add(Conv1D(filters=2, kernel_size=2, activation='relu'))
    model.add(GlobalMaxPool1D())
    #model.add(Concatenate())
    model.add(Dense(len(categories), activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    plot_model(model, to_file='base_model.png', show_shapes=True)
    model.summary()
    return model


def linear_model_combined():

    # declare input
    inlayer =Input(shape=(max_length,))
    #flatten = Flatten()(inlayer)
    embedding_layer = Embedding(num_words,300, embeddings_initializer=Constant(embedding_matrix),trainable=False)(inlayer)

    modela = Conv1D(filters=32, kernel_size=2)(embedding_layer)
    modela = Activation('relu')(modela)
    modela = GlobalMaxPool1D()(modela)

    modelb = Conv1D(filters=32, kernel_size=3)(embedding_layer)
    modelb = Activation('relu')(modelb)
    modelb = GlobalMaxPool1D()(modelb)
    added = Concatenate()([modela,modelb])
    out = Dense(4,activation='softmax')(added)

    #model_concat = Activation('relu')(model_concat)
    #model_concat = Dense(256)(model_concat)
    #model_concat = Activation('relu')(model_concat)

    #model_concat = Dense(4)(model_concat)
    #model_concat = Activation('softmax')(model_concat)

    #model_combined = Model(inputs=inlayer,outputs=model_concat)
    #model_combined = Sequential(added)
    #model_combined.add(out)
    model_combined = Model(inputs = inlayer,outputs = out)
    model_combined.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',util.precision_m,util.recall_m,util.f1_m])
    model_combined.summary()
    #merged = Model(inputs=[], outputs=[A3, B3])
    #plot_model(model_combined, to_file='demo.png', show_shapes=True)
    return model_combined


mod = linear_model_combined()
train_x =padded_doc[:10000]
test_x = padded_doc[10001:]
train_y = np.asarray(Y[:10000])
encoded_train_y = np_utils.to_categorical(train_y)
test_y = np.asarray(Y[10001:])
encoded_test_y = np_utils.to_categorical(test_y)
mod.fit(train_x,encoded_train_y,epochs=20, batch_size=10)
accuracy =mod.evaluate(test_x,encoded_test_y)
print(accuracy)