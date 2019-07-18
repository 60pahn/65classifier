import numpy as np
import sklearn as sk  
import pandas as pd
import os
import seaborn as sns

import tensorflow as tf

import matplotlib.pyplot as plt
#standardizing the input feature
from sklearn.preprocessing import StandardScaler
#data splitter
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

from sklearn.metrics import confusion_matrix

sns.set(style="darkgrid")


#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix 
#from sklearn import svm
#from sklearn.svm import SVC

#change data directory to DATA folder
os.chdir('D:/Python/Data')
a = os.getcwd()
print("Data folder is: "+ a)

#define a variable for our data from a file in the curreent dir
dataset = pd.read_csv('pima-indians-diabetes.data.csv', header=0)
#print the 6 first rows of data
print(dataset.head(n=6))
print(dataset.describe(include='all'))

#selecting last column of data frame and all rows
y = dataset.iloc[:,8] 
#selecting 1st to 8th column of data frame and all rows
X = dataset.iloc[:,0:8]

print(X.head(n=3))


#sns.heatmap(dataset.corr(), annot=True)

sc = StandardScaler()
X = sc.fit_transform(X)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

###model = Sequential()
###model.add(Dense(64, input_dim=8, init='uniform', activation='relu'))
###model.add(Dropout(0.5))
###model.add(Dense(64, activation='relu'))
###model.add(Dropout(0.5))
###model.add(Dense(1, activation='sigmoid'))

###model.compile(loss='binary_crossentropy',
###              optimizer='rmsprop')
###eval_model=model.evaluate(X_train, y_train)



model = Sequential()
#First Hidden Layer
model.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8))
#Second  Hidden Layer
model.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
#Compiling the neural network
model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
#Fitting the data to the training dataset
model.fit(X_train,y_train, batch_size=10, epochs=300)
eval_model=model.evaluate(X_train, y_train)
eval_model

y_pred=model.predict(X_test)
y_pred =(y_pred>0.5)

cm = confusion_matrix(y_test, y_pred)
print("CM is:")
print(cm)
print("Eval is:")
print(eval_model)

plt.show()

