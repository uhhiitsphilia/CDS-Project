#!/usr/bin/env python
# coding: utf-8

# In[4]:


# 1. Load the IMDB dataset using pandas
# 2. Pre-process the dataset by removing html tags, punctuations and numbers, multiple spaces
# 3. Convert the text to sequences. You may need the following steps

# tokenizer = Tokenizer(num_words=5000) #Define a Keras tokenizer;
# from keras.preprocessing.text import Tokenizer

# tokenizer.fit_on_texts(X_train) # Fit the tokenizer on the text

# X_train = tokenizer.texts_to_sequences(X_train) # Convert the text to sequences
# X_test = tokenizer.texts_to_sequences(X_test)

# 4. Classify the review into positive and negative sentiment categories. You may consider positive class as 1 and negative as 0. Use batch-size 128, optimizer - adam, learning rate - anything, validation split - 0.2, test data split - 0.2, epochs - anything, early_stopping - 10
# The model should contain following layers

# 4.1 -> A trainable Embedding layer with embedding size 100
# 4.2 -> A Dense layer on the embedding layer of output size 128. Add an non linear activation function to the layer. You can use either Relu or tanh.
# 4.3 -> A sigmoid layer for final classification

# 5. Evaluate on the test data and print accuracy.

# 6. Print the model summary and model image


# In[1]:


#import library or frame
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re

# import keras
from keras.layers import embeddings
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Flatten, Input, Activation, LSTM, Bidirectional
from keras.datasets import imdb
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model

#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve,auc

# import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm


# In[2]:


# 1. Load the IMDB dataset using pandas
dataset = pd.read_csv("IMDB Dataset.csv")


# In[3]:


dataset.head()


# In[4]:


# 2. Pre-process the dataset by removing html tags,
#    punctuations and numbers, multiple spaces
# dataset  =  dataset.replace('(<.*?>)','', regex=True)
# dataset  =  dataset.replace(' +',' ', regex=True)
# dataset  =  dataset.replace('[0-9]','', regex=True)
# dataset  =  dataset.replace('[^\w\s]','', regex=True)
# Optional 4: we can add stopwords
# ps=PorterStemmer()
wnl = WordNetLemmatizer()
stopwords=set(stopwords.words('english'))
# Define function for data mining
def normalize_reviews(review):
    #Excluding html tags
    data_tags=re.sub(r'<[^<>]+>'," ",review)
    #Remove special characters/whitespaces
    data_special=re.sub(r'[^a-zA-Z0-9\s]','',data_tags)
    #converting to lower case
    data_lowercase=data_special.lower()
    #tokenize review data
    data_split=data_lowercase.split()
    #Removing stop words
    meaningful_words=[w for w in data_split if not w in stopwords]
    #Appply stemming
    #text= ' '.join([ps.stem(word) for word in meaningful_words])
    #Apply lemmatizing
    text= ' '.join([wnl.lemmatize(word) for word in meaningful_words])
    return text


# In[7]:


#Normalize the train & test data
norm_reviews=dataset['review'].apply(normalize_reviews)


# In[8]:


norm_reviews.head()


# In[9]:


# 4. Classify the review into positive and negative sentiment categories. 
#    positive class as 1 and negative as 0. 
dataset['sentiment']=pd.Categorical(dataset['sentiment'])
dataset['sentiment']=dataset['sentiment'].cat.codes
print(dataset.head())


# In[10]:


# 4*.  test data split - 0.2
X_train, X_test, y_train, y_test = train_test_split(norm_reviews.values, dataset['sentiment'].values,test_size = 0.2)


# In[11]:


# 3. Convert the text to sequences
maxlen = 100
max_features = 5000 # max review length
batch_size = 128
embedding_dim = 128

tokenizer = Tokenizer(num_words=5000) #Define a Keras tokenizer;
tokenizer.fit_on_texts(X_train) # Fit the tokenizer on the text
X_train = tokenizer.texts_to_sequences(X_train) # Convert the text to sequences
X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
X_train = np.array(X_train)
X_test = np.array(X_test)
print(X_train.shape)
# print(X_train[:5])


# In[12]:


print("Building models")
# 4*. Use batch-size 128, optimizer - adam, learning rate - anything, 
# epochs - anything, early_stopping - 10

# 4.1 -> A trainable Embedding layer with embedding size 100
# 4.2 -> A Dense layer on the embedding layer of output size 128. Add an non linear activation function to the layer. You can use either Relu or tanh.
# 4.3 -> A sigmoid layer for final classification
model = Sequential()

model.add(Embedding(max_features, embedding_dim,input_shape=(maxlen,),trainable=True))
model.add(Bidirectional(LSTM(64)))
# model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

plot_model(model, to_file="model.png")
es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



# In[13]:


print("Training ...")
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=10,validation_split=0.2, callbacks=[es])
score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)    


# In[14]:


y_pred = model.predict(X_test)
y_pred = y_pred.ravel()
y_pred = (y_pred>0.5)
y_pred[:10]


# In[15]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[16]:


cr = classification_report(y_test,y_pred)
print('The classification report is: \n',cr)


# In[17]:


#ROC curve for LSTM
fpr,tpr,thresold=roc_curve(y_test,y_pred)
#AUC score for RNN
auc=auc(fpr,tpr)
print('AUC score for RNN with LSTM ::',np.round(auc,3))


# In[18]:


#PLOT for accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[19]:


model.save_weights('BiLSTM_lemma.h5')


# In[ ]:




