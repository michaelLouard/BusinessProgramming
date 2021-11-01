# Section 1. Making Necessary Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

"""Final project for Introduction to Business Programming
    Students: Michael Louard and Grace Barry
    Professor: Vladislav
    Last Updated 10/29/2021
"""

"""The goal of this assignment will be to detect the accuracy of datasets of news
    articles and headlines. Additionally, this program will be able to read tweets, 
    facebook posts, and reddit posts. The encoder used is BERT (Google), 
    and the Machine Learning Tool used will be PyCaret."
"""

# Section 2. Reading Data into dataframe with pandas

newsFrame = pd.read_csv('C:/Users/Louar/Desktop/CWRU/Fall_2021/businessprogramming/final_project/news.csv')

# Getting shape and head of csv file
newsFrame.shape
newsFrame.head()

labels = newsFrame.label
labels.head()

# Checking the balance of the data
plt.pie(label_size, explode=[0.1,0.1],colors=['firebrick','navy'],startangle=90,shadow=True,labels=['Fake','True'],autopct='%1.1f%%')
# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(newsFrame['text'],
                                                    labels, test_size=0.2, random_state=7)

"""TfidVectorizer: Term Frequency Inverse Document Frequency
    This transforms the text from the articles in our csv into meaningful numbers
    that can be interpreted by a Machine Learning tool.
    
    TF-IDF measures the originality of a word by comparing the number of times a word
    appears in a document with the number of documents the word appears in. 
    
    TF-IDF = TF(t,d) * IDF(t)
    IDF = log[(1+n)/(1+df(d,t))] , where n = number of documents & df( = document frequency
"""

# Adding a TfidfVectorizer, if a word has a document frequency higher than 0.7 it gets discarded

tfidf_Vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fitting / Transforming the 'train set, transform test set'

tfidf_train = tfidf_Vectorizer.fit_transform(x_train)
tfidf_test = tfidf_Vectorizer.transform(x_test)

"""PassiveAggressiveClassifier : 
    Classification (online-learning) Algorithm
        Online-Learning : input data comes in sequential order and 
        the machine learning model is updated step-by-step, unlike 
        batch learning, where the entire training dataset is used at once.
    Online-Learning Algorithms are good for large amounts of data 
    
    Passive: If the prediction is correct, keep the model and do not make any changes. 
        i.e., the data in the example is not enough to cause any changes in the model. 
    Aggressive: If the prediction is incorrect, make changes to the model. 
        i.e., some change to the model may correct it.
    
    Initialize w (weight vector)
    monitor stream of articles: 
        receive next document d = (d1...dv)
        apply td.idf, normalize ||d|| = 1
        predict positive if d^t*w > 0 
        observe true class: y = +- 1
        want to enforce: 
            (d^t)*w >= +1 if positive (y=+1)
            (d^t)*w =< if negative (y=-1)
        same as: y((d^t)*w) >= 1
        loss: L = max (0, 1-y((d^t)*w))
        update: w(new) = w + y*L*d
"""
# Initialize the PassiveAggressiveClassifier

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f'Accuracy: {round(score*100,2)}%')

# Confusion Matrix to visualize results

confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])


