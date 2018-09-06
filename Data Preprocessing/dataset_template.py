#set working directory
import os
os.chdir('C:\\Users\\Pato\\Documents\\Machine_Learning_AZ\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------')

#Data Preprocessing

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv("Data.csv")
#make a matrix X out of independet variables
X = dataset.iloc[:, :-1].values
# in [:,:-1] the first : takes all rows, and :-1 takes all but the last one
#make a dependent variable vector
y = dataset.iloc[:,3].values

#optinal display
#np.set_printoptions(threshold = np.nan)

#take care of missing data: use the mean to fill empty values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
#we use 1:3 to access 1 and 2 since the last one is excluded
#replace data
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding cathegorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#thwy are encoded, now we assign them to the first column to replace 
#the strings for the numbers
X[:,0] =labelencoder_X.fit_transform(X[:,0])
#now they are assigned from 0 to 2, but this do not represent their value, 
#none of these means bigger or better than others with lower values
#thus, we do a dummy encoding, separating all categories into columns
#for example, rather than having 1 column of 3 countries
#we should have 3 columns, one for each country, filled with 0s and 1s
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#now we encode the result vector
labelencoder_y = LabelEncoder()
y =labelencoder_y.fit_transform(y)
#now we must split the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
