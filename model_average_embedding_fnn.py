import numpy as np
from keras import Sequential
from keras.layers import Dense,MaxPooling1D
from utility_class import Utility
from keras.utils import np_utils
categories = ['com', 'rec', 'religion', 'politics']
util = Utility()
X, Y =util.get_data(categories)
docs = util.preprocessing_dataset(X)
del X
padded_doc,Y,word_index, max_length = util.pad_input(docs, Y)
#word_embedding_file_name = 'word2vector2.txt'
file_name = 'glove.6B.300d.txt'
#avg_emb,emb_mat = util.average_embedding(word_index,file_name)

avg_emb = util.create_average_embedding(file_name,word_index,300)
word_emb_weight = list()
for doc in padded_doc:
    l =[avg_emb[value] for value in doc]
    word_emb_weight.append(l)
num_words = len(word_index)+1

def get_model():
    model = Sequential()
    # embedding_layer = Embedding(num_words,300, embeddings_initializer=Constant(embedding_matrix), input_length=max_length,trainable=False)
    #model.add(embedding_layer)
    model.add(Dense(500, input_dim=max_length, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy',util.precision_m,util.recall_m,util.f1_m])
    model.summary()
    return model

model = get_model()
train_x = np.array(word_emb_weight[:10000])
test_x = np.array(word_emb_weight[10001:])
train_y = np.asarray(Y[:10000])
encoded_train_y = np_utils.to_categorical(train_y)
test_y = np.asarray(Y[10001:])
encoded_test_y = np_utils.to_categorical(test_y)

model.fit(train_x,encoded_train_y,epochs=20, batch_size=10)
result =model.evaluate(test_x,encoded_test_y)
print(result)