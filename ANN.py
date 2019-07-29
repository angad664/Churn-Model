#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:24:44 2019

@author: angadsingh
"""
# business problem - we want to know how many people are leaving the bank and what are the
# chances that other people close to leaving the bank
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

data = pd.read_csv('Churn_Modelling.csv')

X = data.iloc[:,3:13].values
y = data.iloc[:,13].values


# data processing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_country = LabelEncoder()
X[:,1] = labelencoder_country.fit_transform(X[:,1])
labelencoder_gender = LabelEncoder()
X[:,2] = labelencoder_gender.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#split the data
from sklearn.model_selection import train_test_split
X_test, X_train, y_test, y_train = train_test_split(X,y, test_size =0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# import keras library
from keras.models import Sequential
from keras.layers import Dense

#initialize the ann
classifier = Sequential()

#add input layer and hidden layer
# output_dim = 11(input nodes) +1(output node)= 12/2=6
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu' ,
                     input_dim= 11))

# second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu'))
#add output layer - for output_dim we only want 1 output
# we use sigmoid to get probablity output
# we use softmax function if we dealing with dependent variable with more than 2 classes
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation='sigmoid'))

# compiling ANN means applying Stochastic gradient boosting
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fit ann to training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch =100)

# predict the test result
y_pred = classifier.predict(X_test) #  predicts the prob. that custumers leave the bank
# to convert in true or false : true means customer will leave the bank
y_pred = (y_pred > 0.5)
# New single prediction
"""  predict if the customer with following info will leave the bank?
geography = france
credit score = 600
gender= male
age = 40
tenure = 3
balance = 60000
number of products = 2
has credit card = yesq
is active member = yes
estimated salary = 50000       """
# we compare the data and X. it is orderwise. Also, since we scale the above ann we have to scale this
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5) # false- not leaving
# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu' ,
                     input_dim= 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classfier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs= -1)
mean = accuracies.mean()
variance = accuracies.std()

# improving ann
from keras.layers import Dropout
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu' ,
                     input_dim= 11))
classifier.add(Dropout(p = 0.1))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu'))
classifier.add(Dropout(p = 0.1))


# tuning the ann
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential(optimizer)
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu', input_dim= 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation='sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classfier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy' ,
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
