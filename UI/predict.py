#preprocessing method
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import re
import pickle
import tensorflow as tf
# import numpy as np
from keras.models import Sequential
# from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

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
        # model.add(Embedding(max_features, embedding_dim,input_shape=(maxlen,),trainable=True))
        # model.add(Bidirectional(LSTM(64)))
        # model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # # print(model.summary())
        # model.load_weights(self.weights_path)
        model = tf.keras.models.load_model(self.weights_path)
        return model


def sentiment_scores(sentence): 
  
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
  
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
      
    print("Overall sentiment dictionary is : ", sentiment_dict) 
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
    result = None
    # decide sentiment as positive, negative and neutral 
    # if sentiment_dict['compound'] > 0.05 : 
    #     print("Positive") 
    #     result = ["positive",1]
  
    # elif sentiment_dict['compound'] < - 0.05 : 
    #     print("Negative") 
    #     result = ["negative",0]

    # else : 
    #     print("Neutral") 
    #     result = ["netural",0.5]
    return sentiment_dict

if __name__=="__main__":
    # review = "I am SJ."
    review = "What a magical movie!!! The colors!! The costumes!!! The singing!!! The sets!!!! The magic!!! It's a cross between The Wiz and Greatest Showman. I enjoyed every minute and will surely watch Jingle Jangle every Christmas holiday going forward right alongside Polar Express and Elf."
    tnz_path = 'tnz_raw.pickle'
    weights_path = ['best_model02.hdf5','best_model21.hdf5']
    mode = ["Raw","Raw"]
    y_pred=[]
    for i in range(2):
        mm = multi_models(tnz_path, weights_path[i], mode[i])
        y_pred.append(str(round(mm.postive_score(review),3)))
    print(y_pred)
    
