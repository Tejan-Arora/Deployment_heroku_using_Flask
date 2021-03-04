# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:44:45 2021

@author: tejan
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


dataset = pd.read_csv('C:/Users/tejan/Desktop/Data Analystics/class work_python codes mix/hiring.csv')

dataset.rename({'test_score(out of 10)': 'test_score', 
                'interview_score(out of 10)': 'interview_score', 'salary($)' : 'salary'}, inplace = True, axis = 1)

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)




# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print('salary is ', model.predict([[2, 5, 6]]))

















