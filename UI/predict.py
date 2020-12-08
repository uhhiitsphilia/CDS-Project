#preprocessing method
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import re
import pickle

from keras.models import Sequential
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional

class multi_models():
    def __init__(self, tnz_path, weights_path, mode, other_params=(100,5000,128)):
        self.tnz_path = tnz_path
        self.weights_path = weights_path
        self.mode = mode
        self.other_params = other_params
        self.predict = None

    def postive_score(self, review):
        text = self.normalize_reviews(review)
        X_test = self.token_text(review)
        model = self.build_model()
        y_pred = model.predict(X_test)
        return y_pred[0][0]

    def normalize_reviews(self, review):
        #Excluding html tags
        data_tags=re.sub(r'<[^<>]+>'," ",review)
        #Remove special characters/whitespaces
        data_special=re.sub(r'[^a-zA-Z0-9\s]','',data_tags)
        #converting to lower case
        data_lowercase=data_special.lower()
        #tokenize review data
        data_split=data_lowercase.split()
        #Removing stop words
        en_stopwords=set(stopwords.words('english'))
        meaningful_words=[w for w in data_split if not w in en_stopwords]
        if self.mode == "PorterStemmer":
            ps=PorterStemmer()
            text= ' '.join([ps.stem(word) for word in meaningful_words])
        if self.mode == "WordNetLemmatizer":
            wnl = WordNetLemmatizer()
            text= ' '.join([wnl.lemmatize(word) for word in meaningful_words])
        else:
            text = meaningful_words
        return text

    def token_text(self, review):
        (maxlen, max_features, embedding_dim)= self.other_params
        tokenizer = None
        with open(self.tnz_path,'rb') as handle:
            tokenizer = pickle.load(handle)
        X_test = tokenizer.texts_to_sequences([review])
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        return X_test
    
    def build_model(self):
        (maxlen, max_features, embedding_dim)= self.other_params
        model = Sequential()
        model.add(Embedding(max_features, embedding_dim,input_shape=(maxlen,),trainable=True))
        model.add(Bidirectional(LSTM(64)))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # print(model.summary())
        model.load_weights(self.weights_path)
        return model
    
if __name__=="__main__":
    tnz_path = ['tnz_stem.pickle','tnz_lemm.pickle']
    weights_path = ['BiLSTM_stem.h5','BiLSTM_lemm.h5']
    mode = ["PorterStemmer","WordNetLemmatizer"]
    review = "Grab the family, get comfy & enjoy great singing, acting, sets, & animation. The people that rated this low apparently need more movies like this in their lives. It was beautiful, clean and added a bit of magic & belief which is what we all need this Christmas. Wonderfully over the top - I highly recommend watching!!"
    for i in range(2):
        mm = multi_models(tnz_path[i], weights_path[i], mode[i])
        y_pred = mm.postive_score(review)
        print(y_pred)
    
