# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:23:39 2019

@author: kashi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')
dataset['Review'][0]

clean_Review=[]

for i in range(1000):
    #getting rid of unwanted character
    Review=re.sub('@[\w]*',' ',dataset['Review'][i])
    Review=re.sub('[^a-zA-Z#]',' ',Review)
    Review=Review.lower()
    #splitting every word from sentence
    Review=Review.split()
    #temp=[token for token in tweet if not token in stopwords.words]
    # iterating through each words and checking if they are stopwords or not
  # if they are stopwords we will not consider them furthermore
    Review=[ps.stem(token) for token in Review if not token in stopwords.words('english')]
    Review=' '.join(Review)
    clean_Review.append(Review)

# creating the bag of words model    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(clean_Review)
X=X.toarray()
#y=dataset['Liked'].values

print(cv.get_feature_names())

# reducing the number of features

# creating the count vectorizer model with max_features
cv = CountVectorizer(max_features = 1200)

# feeding the corpus data to the count vectorizer model
X = cv.fit_transform(clean_Review).toarray()

# checking the shape
print(X.shape)

# maing the dependent variable column
y = dataset.iloc[:, 1].values
print(y.shape)

# splitting the dataset into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# min max scaling

from sklearn.preprocessing import MinMaxScaler

# creating a min max scaler
mm = MinMaxScaler()

# feeding the independent variables into the model
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

# checking the accuracies
print("Training Accuracy :", gnb.score(X_train, y_train))
print("Testing Accuracy :", gnb.score(X_test, y_test))

# making the confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix (y_test, y_pred)
print(cm)

X_new=([['food is soo delicious,loved it']])
X_new1=gnb.fit(X_new,y)
y_predict=gnb.predict(X_new)


