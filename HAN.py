import gc
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import sys
import os
import pickle

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from keras.layers import Embedding, BatchNormalization
from keras.layers import Dense, Input, Flatten
from keras.layers import Embedding, Dropout, GRU, Bidirectional, TimeDistributed, SimpleRNN, LSTM
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K, initializers, regularizers, constraints
from keras import initializers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras.utils import CustomObjectScope
from keras.engine.topology import Layer
from keras.initializers import Constant
from alt_model_checkpoint import AltModelCheckpoint
from keras.callbacks import TensorBoard
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.utils import shuffle
import seaborn as sns
import unicodedata
import nltk
from nltk import tokenize
from nltk.corpus import stopwords 
from nltk.tokenize.regexp import RegexpTokenizer
from confusion_matrix_pretty_print import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class AttLayer(Layer):
    def __init__(self, attention_dim,**kwargs):
        self.init = initializers.get('normal')
        self.regularizer=regularizers.l2(1e-4)
        self.attention_dim = attention_dim
        
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3 ###입력으로 들어온 input shape가 3차원 형태여야함, 검사 중
        
        self.W = K.variable(self.init(shape=(input_shape[-1], self.attention_dim)), name='{}_W'.format(self.name))
        self.b = K.variable(self.init(shape=(self.attention_dim, )), name='{}_b'.format(self.name))
        self.u = K.variable(self.init(shape=(self.attention_dim, 1)), name='{}_u'.format(self.name))


        # self.W = self.add_weight(shape=(input_shape[-1], self.attention_dim),
        #                         name='{}_W'.format(self.name),)
        #                          # initializer=self.init,
        #                          # regularizer=self.regularizer,
        # self.b = self.add_weight(shape=(self.attention_dim, ),
        #                         name='{}_b'.format(self.name),)
        #                          # initializer='zero',
        #                          # regularizer=self.regularizer,
        # self.u = self.add_weight(shape=(self.attention_dim, 1),
        #                         name='{}_u'.format(self.name),)
        #                          # initializer=self.init,
        #                          # regularizer=self.regularizer,

        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.exp(K.squeeze(K.dot(uit, self.u), axis=-1))

        if mask is not None:
#      Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
            
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output
    
    def get_config(self):
        config = super().get_config()
        config['attention_dim'] = self.attention_dim
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Hierarchical_attention_networks():
    def __init__(self, TRAINING_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, 
        MAX_SEQUENCE_LENGTH, MAX_SENTS, K_num):

        self.TRAINING_SPLIT = TRAINING_SPLIT
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.TEST_SPLIT = TEST_SPLIT
    
        
        self.EMBEDDING_DIM = 300
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.batch_size = 128
        self.epochs = 20
        self.l2_reg = regularizers.l2(1e-4)
        self.K_num = K_num

        self.MAX_SENTS = MAX_SENTS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.MAX_NB_WORDS = 50000
        
        self.file_name = None
        self.Result_path = None
        self.Model_path = None

        self.category = None
        self.text_type = None
        self.date_type = None
        self.data_type = None
    
        self.embedding_matrix = None
        self.embedding_num_words = None
        
        ##load_weight
        self.base_model = None
        self.word_attention_model = None
        self.tokenizer_state = None
        self.tokenizer = None
        self.stop_words = set(stopwords.words('english')) 

        self.train_position = None
        self.test_position = None
        self.train_position_list = []
        self.test_position_list = []



        ###save_load_option: 1->save, 2->load
    def fold_position_save_load(self, num_fold, save_load_option, train_position_list=None, test_position_list=None):

        if save_load_option == 1:
            f = open(self.Model_path + "/" + self.file_name + "_" + str(num_fold) + "_fold_position_list.txt", "w")

            for element in train_position_list:
                f.write(str(element) +" ")
            f.write("\n")
            for element in test_position_list:
                f.write(str(element) +" ")
            f.close()

        elif save_load_option == 2:
            f = open(self.Model_path + "/" + self.file_name + "_" + str(num_fold) + "_fold_position_list.txt", "r")

            self.train_position = f.readline().strip()
            self.train_position_list = self.train_position.split(" ")
            self.train_position = np.array(self.train_position_list)
            self.train_position = self.train_position.astype(int)

            self.test_position = f.readline().strip()
            self.test_position_list = self.test_position.split(" ")
            self.test_position = np.array(self.test_position_list)
            self.test_position = self.test_position.astype(int)

        return self.train_position, self.test_position
        
    def _save_tokenizer_on_epoch_end(self, epoch, num_fold):
        if epoch == 0:
            self.tokenizer_state = {'tokenizer':self.tokenizer}
            pickle.dump(self.tokenizer_state, open(self.Model_path + "/" + self.file_name + "_" + str(num_fold) + ".tokenizer", "wb"))
            
    def _normalized_text(self, text):
        text = unicodedata.normalize("NFKD", text)
        text = text.strip().lower()
        sentences_token = tokenize.sent_tokenize(text)
        # sentences = self.rm_stopwords(sentences_token)
        sentences = sentences_token

        return sentences

    def _encoded_text(self, x):

        x = np.array(x)
        if not x.shape:
            x = np.expand_dims(x, 0)
        texts = np.array([self._normalized_text(text) for text in x])

        encoded_texts = np.zeros((len(texts), self.MAX_SENTS, self.MAX_SEQUENCE_LENGTH), dtype='int32')
        for i, text in enumerate(texts):
            encoded_text = np.array(pad_sequences(self.tokenizer.texts_to_sequences(text), maxlen=self.MAX_SEQUENCE_LENGTH, padding='post'))[:self.MAX_SENTS]
            encoded_texts[i][:len(encoded_text)] = encoded_text

        return encoded_texts
    
    def data_read(self, data_type ,category, text_type):

        if os.path.isdir("./Results") == False:
            os.mkdir("./Results")
        
        if (os.path.isdir("./Results/result") == False) and (os.path.isdir("./Results/model") == False):
            os.mkdir("./Results/result")
            os.mkdir("./Results/model")

        self.Result_path = "./Results/result/" + "B:" + str(self.batch_size) + "_D:" + str(self.dropout) + "_L:" + str(self.learning_rate) + "_Dim:" + str(self.EMBEDDING_DIM) + "_E:" + str(self.epochs) + "_MaxWords:" + str(self.MAX_NB_WORDS) +"_MaxWordLength:" + str(self.MAX_SEQUENCE_LENGTH) + "_SenLength:" + str(self.MAX_SENTS)
        self.Model_path = "./Results/model/" + "B:" + str(self.batch_size) + "_D:" + str(self.dropout) + "_L:" + str(self.learning_rate) + "_Dim:" + str(self.EMBEDDING_DIM) + "_E:" + str(self.epochs) + "_MaxWords:" + str(self.MAX_NB_WORDS) +"_MaxWordLength:" + str(self.MAX_SEQUENCE_LENGTH) + "_SenLength:" + str(self.MAX_SENTS)

        if os.path.isdir(self.Result_path) == False:
            os.mkdir(self.Result_path)
        if os.path.isdir(self.Model_path) == False:
            os.mkdir(self.Model_path)

        self.date_type = ["0"]
        self.date_type = self.date_type[0]

        self.Result_path = self.Result_path + "/" + str(self.date_type)
        self.Model_path = self.Model_path + "/" + str(self.date_type)

        if os.path.isdir(self.Result_path) == False:
            os.mkdir(self.Result_path)
        if os.path.isdir(self.Model_path) == False:
            os.mkdir(self.Model_path)
        
        self.data_type = data_type
        self.category = category
        self.text_type = text_type
        
        file = self.data_type + "_" + self.category + "_" + self.text_type + "_" + str(self.date_type) + ".tsv"
        print (file)
        self.file_name = file.split(".")[0]
        crowdfunding_data = pd.read_csv("./data/"+file, sep='\t',encoding="utf-8", error_bad_lines = False)
        
        return crowdfunding_data


    def rm_stopwords(self, sentences_token):

        reconsturcted_sentences = []
        for sent in sentences_token:
            reconsturcted_sentences.append(' '.join(w for w in nltk.word_tokenize(sent) if w.lower() not in self.stop_words))

        return reconsturcted_sentences
        

    def text_preprocessing(self, data):
    
        texts = []
        fit_on_texts_sentences = []
        labels = []
        project_texts = []

        for idx in range(data.content.shape[0]):        
            text = unicodedata.normalize("NFKD", str(data.content[idx]))
            text = text.strip().lower()
            text = text.replace("<h1>", "").replace("</h1>","")
            text = text.replace("you'll need an html5 capable browser to see this content. play replay with sound play with sound 00:00 00:00", "")
            texts.append(text)
            sentences_token = tokenize.sent_tokenize(text) ## 문서에서 문장을 구분 함
            # sentences = self.rm_stopwords(sentences_token)
            sentences = sentences_token
            project_texts.append(sentences)
            labels.append(data.success_or_fail[idx])

            fit_on_texts_sentences.append(' '.join(sentences))

        self.tokenizer = Tokenizer(oov_token="UNK", num_words=self.MAX_NB_WORDS+1)
        self.tokenizer.fit_on_texts(fit_on_texts_sentences)
        self.word_index = self.tokenizer.word_index

        data = np.zeros((len(texts), self.MAX_SENTS, self.MAX_SEQUENCE_LENGTH), dtype='int32')
        for i, sentences in enumerate(project_texts):
            for j, sent in enumerate(sentences):
                if j < self.MAX_SENTS:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        if k<self.MAX_SEQUENCE_LENGTH and self.word_index[word] < self.MAX_NB_WORDS:
                            data[i,j,k] = self.word_index[word]
                            k = k + 1

        labels = np.array(labels)

        data, labels = shuffle(data, labels)

    #   for i, sentences in enumerate(project_texts):
    #     if len(sentences) < MAX_SENTS:
    #      k = 0
    #      for j in range(len(sentences),MAX_SENTS):   
    #        data[i][j] = data[i][k]
    #        k = k + 1
    #        if k == len(sentences):
    #         k = 0

        print('Found %s unique tokens.' % len(self.word_index))

        return data, labels, texts
    
    def data_split(self, data, labels):

        ## 1 = success, 0 = fail
        labels = np.asarray(labels).astype('int32')
        
        class1_indices = np.where(labels == 1)
        class2_indices = np.where(labels == 0)
        
        x_1 = data[class1_indices]
        y_1 = labels[class1_indices]
        x_0 = data[class2_indices]
        y_0 = labels[class2_indices]
        
        class1_indices = np.array(class1_indices[0])
        class2_indices = np.array(class2_indices[0])
        
        np.random.seed(20)
        np.random.shuffle(class1_indices)
        np.random.shuffle(class2_indices)

        n = min(len(class1_indices), len(class2_indices))
        train_num = int(round(self.TRAINING_SPLIT *n))
        valid_num = int(round(self.VALIDATION_SPLIT *n))
        test_num = int(round(self.TEST_SPLIT *n))
        
        train_x = np.concatenate([x_1[:train_num], x_0[:train_num]])
        train_y = np.concatenate([y_1[:train_num], y_0[:train_num]])
        valid_x = np.concatenate([x_1[train_num:train_num+valid_num], x_0[train_num:train_num+valid_num]])
        valid_y = np.concatenate([y_1[train_num:train_num+valid_num], y_0[train_num:train_num+valid_num]])
        test_x = np.concatenate([x_1[train_num+valid_num:], x_0[train_num+valid_num:]])
        test_y = np.concatenate([y_1[train_num+valid_num:], y_0[train_num+valid_num:]])

        ori_y = np.concatenate((train_y, valid_y, test_y), axis=0)

        train_y = to_categorical(train_y)
        valid_y = to_categorical(valid_y)
        test_y = to_categorical(test_y)
        
    #     print (np.argmax(train_y, axis=1))

        # print ("Training: %s, Validation: %s, Test: %s" % (len(train_x), len(valid_x) ,len(test_x)))
        # print ("Training: %s, Validation: %s, Test: %s" % (train_y.sum(axis=0), valid_y.sum(axis=0) ,test_y.sum(axis=0)))

        x = np.concatenate((train_x, valid_x, test_x), axis=0)
        y = np.concatenate((train_y, valid_y, test_y), axis=0)


        kfold = StratifiedKFold(n_splits=self.K_num, shuffle=True)
        return (x, y, ori_y, kfold)

        # return (train_x, train_y), (valid_x, valid_y), (test_x ,test_y)
    
    
    def make_embedding_layer(self):

        GLOVE_DIR = "../Glove_dict/"
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'crawl-300d-2M.vec'), encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        ## give vector to word
        self.embedding_num_words = min(self.MAX_NB_WORDS, len(self.word_index)+1)
        self.embedding_matrix = np.random.random((self.embedding_num_words, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if i < self.MAX_NB_WORDS:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector

    
    def HAN_layer(self):

        self.base_model = None
        self.gpu_model = None

        sentence_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_word_sentence = Embedding(self.embedding_num_words,
                                           self.EMBEDDING_DIM,
                                           embeddings_initializer=Constant(self.embedding_matrix),
                                           input_length=self.MAX_SEQUENCE_LENGTH,
                                           trainable=True,
                                           mask_zero=True,
                                           name='word_embedding')(sentence_input)

        embedded_sentence = Dropout(self.dropout)(embedded_word_sentence)
        word_encoder = Bidirectional(GRU(150, return_sequences=True, kernel_regularizer=self.l2_reg))(embedded_sentence)
        word_dense_layer = Dense(300, activation='relu', name='word_dense_w')(word_encoder)
        word_attention_layer = AttLayer(300, name='word_attention_layer')(word_dense_layer)
        # dropout_word_attention = Dropout(self.dropout)(word_attention_layer)
        word_attention = Model(inputs = sentence_input, outputs = word_attention_layer)
        word_attention.summary()

        document_input = Input(shape=(self.MAX_SENTS, self.MAX_SEQUENCE_LENGTH), dtype='int32')
        time_document_input = TimeDistributed(word_attention)(document_input)
        sentence_encoder = Bidirectional(GRU(150, return_sequences=True, kernel_regularizer=self.l2_reg))(time_document_input)
        sentence_dense_layer = Dense(300, activation='relu', name='sentence_dense_w')(sentence_encoder)
        sentence_attention_layer = AttLayer(300, name='sentence_attention_layer')(sentence_dense_layer)
        # dropout_sentence_attention = Dropout(self.dropout)(sentence_attention_layer)

        fc_layers = Sequential()
        # fc_layers.add(Dense(3, activation='softmax'))
        # preds = Dense(3, activation='softmax')(dropout_sentence_attention)
        fc_layers.add(Dense(300, activation='relu'))
        # fc_layers.add(BatchNormalization())
        fc_layers.add(Dropout(self.dropout))
        fc_layers.add(Dense(2, activation='softmax'))

        preds = fc_layers(sentence_attention_layer)
        self.base_model = Model(document_input, preds)
        self.base_model.summary()

        # self.gpu_model = multi_gpu_model(self.base_model, gpus=2)
        self.base_model.compile(loss='categorical_crossentropy', 
                      # optimizer=RMSprop(lr=self.learning_rate,rho=0.9, epsilon=None, decay = 1e-4),
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['acc']
                     )
        
    def training(self, x, y, train, num_fold):

        hist = self.base_model.fit(x[train], y[train], validation_split=0.11, verbose=1, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                         callbacks = [
                             # AltModelCheckpoint(filepath=(self.Model_path + "/" + self.file_name + "_" + str(num_fold) + ".h5"), alternate_model=self.base_model ,monitor='val_loss', save_best_only=True),
                             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5),
                             LambdaCallback(on_epoch_end=lambda epoch, logs: self._save_tokenizer_on_epoch_end(epoch, num_fold)),
                             # TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True),
                             EarlyStopping(patience=3, monitor='val_loss')
                             ]
                        )
        return hist
    
    def train_progress(self, x, y, test, hist, num_fold, acc_per_fold):
        y_pred = self.base_model.predict(x[test], batch_size=self.batch_size)
        matrix = confusion_matrix(y[test].argmax(axis=1), y_pred.argmax(axis=1))
        report = classification_report(y[test].argmax(axis=1), y_pred.argmax(axis=1))

        TN = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TP = matrix[1][1]

        success_precision = TN / (TN + FN)
        success_recall = TN / (TN + FP)
        fail_precision = TP / (TP + FP)
        fail_recall = TP / (TP + FN)
        total_accuracy = (TN + TP) / (TN + FP + FN + TP)

        h = open(self.Result_path + "/" + self.file_name + "_" + str(num_fold) + "_result.txt", "w")

        h.write("loss\tacc\tval_loss\tval_acc\n")
        for num in range(0,len(hist.history['loss'])):
            h.write(str(hist.history['loss'][num]) + "\t" + str(hist.history["acc"][num]) + "\t" + str(hist.history["val_loss"][num]) + "\t" + str(hist.history["val_acc"][num]) + "\n")
        h.write("\n")
        h.write("test_acc " + str(total_accuracy) + "\n")
        h.write("success_precision: " + str(success_precision) + "\tsuccess_recall: " + str(success_recall) + "\nfail_precision: " +
               str(fail_precision) + "\tfail_recall: " + str(fail_recall))
        if num_fold == self.K_num:
            h.write("\n\ntotal_accuracy: " + str(np.mean(acc_per_fold)))
        h.close()

        fig, loss_ax = plt.subplots()

        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

        acc_ax.plot(hist.history['acc'], 'b', label='train acc')
        acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.savefig(self.Result_path + "/" + self.file_name + "_" + str(num_fold) + ".png")

        acc_per_fold.append(total_accuracy)
        return acc_per_fold
        # matrix_array = np.array(matrix)
        # df_cm = DataFrame(matrix_array, index=range(1,3), columns=range(1,3))
        # cmap='GnBu_r'
        # pretty_plot_confusion_matrix(df_cm, save_loc=self.Result_path, file_name=self.file_name, cmap=cmap)

        
    def load_weights(self, num_fold):
        self.base_model = load_model(self.Model_path + "/" + self.file_name + "_" + str(num_fold) + ".h5", custom_objects={'AttLayer':AttLayer})
        time_distributed_name = "time_distributed_" + str(num_fold)
        self.word_attention_model = self.base_model.get_layer(time_distributed_name).layer
        self.tokenizer_state = pickle.load(open(self.Model_path + "/" + self.file_name + "_" + str(num_fold) + ".tokenizer", "rb" ))
        self.tokenizer = self.tokenizer_state['tokenizer']
        
    def activation_maps(self, project_sentences):
        def softmax(a):
            return np.exp(a) / np.sum(np.exp(a))

        reverse_word_index = {value:key for key,value in self.word_index.items()}

        normalized_text = self._normalized_text(project_sentences)
        encoded_text = self._encoded_text(project_sentences)[0]

        ####get word attention
        word_encoding_out = Model(inputs = self.word_attention_model.input, outputs = self.word_attention_model.get_layer('word_dense_w').output)
        word_encodings = word_encoding_out.predict(encoded_text)
        word_context = self.word_attention_model.get_layer('word_attention_layer').get_weights()[2] ##context vector
        u_wattention = encoded_text * np.exp(np.squeeze(np.dot(word_encodings, word_context)))

        nopad_encoded_text = encoded_text[:len(normalized_text)]
        nopad_encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text]
        reconstructed_texts = [[reverse_word_index[int(i)] for i in sentence] for sentence in nopad_encoded_text]

        nopad_wattention = u_wattention[:len(normalized_text)]
        nopad_wattention = nopad_wattention / np.expand_dims(np.sum(nopad_wattention, -1), -1)
        nopad_wattention = np.array([softmax(attention_seq[:len(sentence)]) 
                                     for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)])


        text_list = []
        attention_list = []
        word_activation_maps = []
        for i, text in enumerate(reconstructed_texts):
            text_list.append(text)
            attention_list.append(nopad_wattention[i])
            word_activation_maps.append(list(zip(text, nopad_wattention[i])))


        ####get sentence attention 
        sentence_encoding_out = Model(inputs = self.base_model.input, outputs = self.base_model.get_layer('sentence_dense_w').output)
        sentence_encoded_text = np.expand_dims(encoded_text, 0)
        sentence_encoding = np.squeeze(sentence_encoding_out.predict(sentence_encoded_text), 0)
        sentence_context = self.base_model.get_layer('sentence_attention_layer').get_weights()[2] ##context vector
        u_sattention = np.exp(np.squeeze(np.dot(sentence_encoding, sentence_context)))

        nopad_sattention = u_sattention[:len(normalized_text)]
        nopad_sattention = nopad_sattention / np.expand_dims(np.sum(nopad_sattention, -1), -1)


        activation_map = list(zip(word_activation_maps, nopad_sattention))
        return activation_map, text_list, attention_list, nopad_sattention




    def activation_map_distribution(self, x, y, test_texts):

        if os.path.isdir("./attention_result") == False:
            os.mkdir("./attention_result")


        for i, sentences in enumerate(test_texts):

            redemension_text = np.expand_dims(x[i], 0)
            y_pred = self.base_model.predict(redemension_text)[0]
            pred_class = y_pred.argmax(axis=0)
            real_class = y[i].argmax(axis=0)
            

            if os.path.isfile("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "_classification_result.txt") == False:
                f = open("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "_classification_result.txt", "a")
                f.write("num\treal_class\tpred_class\tProb(success)\tProb(fail)\n")
            else:                
                f = open("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "_classification_result.txt", "a")
            f.write(str(self.test_position_list[i]) + "\t" + str(real_class) + "\t" + str(pred_class) + "\t" + str(y_pred[1]) + "\t" + str(y_pred[0]) +"\n")
            f.close()

            if os.path.isdir("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type) == False:
                os.mkdir("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type)
                os.mkdir("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "/1")
                os.mkdir("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "/0")


            result_activation_maps = self.activation_maps(sentences)

            f2 = open("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "/" + str(real_class) + "/" + str(self.test_position_list[i]) + ".txt", "w")
            for maps in result_activation_maps[0]:
                f2.write(str(maps[1]) + "\t")
                for word in maps[0]:
                    f2.write(str(word[0].encode("ascii","ignore").decode("ascii")) + " " + str(word[1]) + "\t")
                f2.write("\n")
            f2.close()

    def activation_visualization(self, x, y, texts):
        if os.path.isdir("./attention_result") == False:
            os.mkdir("./attention_result")
        if os.path.isdir("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "_" + str(self.date_type)) == False:
            os.mkdir("./attention_result/" + self.data_type + "_" +self.category + "_" + self.text_type + "_" + str(self.date_type))
            os.mkdir("./attention_result/" + self.data_type + "_" +self.category + "_" + self.text_type + "_" + str(self.date_type) + "/0")
            os.mkdir("./attention_result/" + self.data_type + "_" +self.category + "_" + self.text_type + "_" + str(self.date_type) + "/1")
        
        
        for i, sentences in enumerate(texts):
            real_class = y[i].argmax(axis=0)

            if os.path.isdir("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "_" + str(self.date_type) + "/" + str(real_class) + "/"+ str(self.test_position_list[i])) == False:
                os.mkdir("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "_" + str(self.date_type) + "/" + str(real_class) + "/"+ str(self.test_position_list[i]))

            attention_result_line = open("./attention_result/" + self.data_type + "_" + self.category + "_" + self.text_type + "/" + str(real_class) + "/" + str(self.test_position_list[i]) + ".txt", "r").readlines()
            for line in attention_result_line:
                line = line.strip().split("\t")
                sattention = line[0]
                
                word_attention_pairs = line[1:]
                word_list = list(pair.split(" ")[0] for pair in word_attention_pairs)
                attention_list = list(float(pair.split(" ")[1]) for pair in word_attention_pairs)
                
                df = pd.DataFrame(attention_list, index=word_list, columns=[str(sattention)]).T

                fig, ax1 = plt.subplots(figsize=(len(attention_list)+2, 10))
                sns.heatmap(df, color="green", vmin=0, vmax=1, annot=True, linewidths=.1, cmap="Reds", fmt=".3f")
                plt.xticks(rotation=45, fontsize=15)
                plt.yticks(rotation=0, fontsize=15)
                plt.savefig("attention_result/" + self.data_type + "_" + self.category + "_"+self.text_type + "_" + str(self.date_type) + "/" +  str(real_class) + "/" + str(self.test_position_list[i]) + "/" + str(sattention) + ".png", bbox_inches='tight')
                plt.close('all')
                
                f = open("attention_result/" + self.data_type + "_" + self.category + "_"+ self.text_type + "_" + str(self.date_type) + "/" +  str(real_class) + "/" + str(self.test_position_list[i]) + "/" + str(sattention) + ".txt", "w")
                for word in word_list:
                    f.write(str(word.encode('ascii', 'ignore'))+"\n")
                f.close()

                
    def clear_model(self):
        del self.base_model
        gc.collect()