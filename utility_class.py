from sklearn.datasets import fetch_20newsgroups
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras import backend as K
#from keras import backend as K
import tensorflow as tf




class Utility:

    def get_data(self,categories):
        nltk.download('punkt')
        categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','rec.autos','rec.motorcycles',
         'rec.sport.baseball','rec.sport.hockey','soc.religion.christian','talk.religion.misc','talk.politics.guns','talk.politics.mideast','talk.politics.misc']

        twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, categories=categories)
        twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42, categories=categories)
        X = twenty_train.data + twenty_test.data
        Y = np.concatenate((twenty_train.target, twenty_test.target), axis=0)
        com = {0, 1, 2, 3, 4}
        rec = {5, 6, 7, 8}
        religion = {9, 10}
        politics = {11, 12, 13}
        for i in range(len(Y)):
            if Y[i] in com:
                Y[i] = 0
            elif Y[i] in rec:
                Y[i] = 1
            elif Y[i] in religion:
                Y[i]= 2
            elif Y[i] in politics:
                Y[i]=3


        return X,Y

    def preprocessing_dataset(self, dataset):
        preprocessed_docs = list()
        for data in dataset:
            tokens = word_tokenize(data)
            # convert to lowercase
            tokens = [word.lower() for word in tokens]
            # remove punctuation from each word
            # table = str.maketrans('','', string.punctuations)
            # filter out stopwords

            # remove tokens that are not alphabet
            tokens = [word for word in tokens if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if not w in stop_words]
            preprocessed_docs.append(tokens)
        return preprocessed_docs



    def pad_input(self, docs,target):
        final_docs =[]
        final_target =[]
        for idx, val in enumerate(docs):
            if len(val)<1001:
                final_docs.append(val)
                final_target.append(target[idx])
        #docs = [doc for doc in docs if len(doc)<1001]
        print('Total Document=',len(final_docs))
        tokenizer_object = tf.keras.preprocessing.text.Tokenizer()
        tokenizer_object.fit_on_texts(final_docs)
        max_length = max([len(doc) for doc in final_docs])
        max_length_index = np.argmax([len(doc) for doc in final_docs])
        total_doc_with_length = [index for index, doc in enumerate(final_docs) if len(doc) > 1000]
        print(total_doc_with_length)
        sequences = tokenizer_object.texts_to_sequences(final_docs)

        # pad sequence
        word_index = tokenizer_object.word_index
        doc_pad = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)
        word_embedding_file_name = 'word2vector2.txt'

        num_words = len(word_index) + 1
        return doc_pad,final_target, word_index, max_length

    def read_wordembedding(self, file_name):
        embedding_index = {}
        f = open(os.path.join('', file_name), encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embedding_index[word] = coefs
        f.close()
        return embedding_index

    def average_embedding(self,word_index, word_embedding_file_name):
        embedding_index = self.read_wordembedding(word_embedding_file_name)
        num_words = len(word_index)+1
        embedding_matrix = np.zeros((num_words, 300))
        weighted_average_emb = {}
        weighted_average_emb[0] = 0
        embedding_word = embedding_index.keys()
        len1 = len(embedding_index)
        len2 = len(word_index)

        for word, i in word_index.items():
            # if i > num_words:
            #   continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                average = np.average(embedding_vector.astype(np.float))
                embedding_matrix[i] = embedding_vector
                weighted_average_emb[i] = average
            else:
                weighted_average_emb[i] = 0
        return weighted_average_emb, embedding_matrix

    def create_embedding_matrix(self,filepath, word_index, embedding_dim):
        vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        with open(filepath, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coeff = np.asarray(values[1:])
                #word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word]
                    embedding_matrix[idx]=coeff
                    #embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

        return embedding_matrix

    def create_average_embedding(self,filepath, word_index, embedding_dim):
        vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
        weighted_average_emb = {val: 0 for key,val in word_index.items()}
        weighted_average_emb[0] = 0

        with open(filepath, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coeff = np.asarray(values[1:])
                #word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word]
                    weighted_average_emb[idx]= np.average(coeff.astype(np.float))
                    #embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]


        return weighted_average_emb


    def recall_m(self,y_true, y_pred):
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
            return recall

    def precision_m(self,y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    def f1_m(self,y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

