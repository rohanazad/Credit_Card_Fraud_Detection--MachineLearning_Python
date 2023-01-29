
"""
Credit Card Fraud Detection

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('C:/Users/rohu1/Desktop/Credit Card Fraud Detection/creditcard.csv')


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()



"""
This Dataset is highly unblanced-

0 = Normal Transaction = 2.84 Lakh
1 = Fraudulent transaction = 492

"""



# separating the data (based on the two labels) for analysis
legit = credit_card_data.loc[(credit_card_data.Class==0)]
fraud = credit_card_data.loc[(credit_card_data.Class==1)]


# prepare a histogram of amount for both the categories
plt.hist(legit.Amount, edgecolor="red", bins=10)
plt.hist(fraud.Amount, edgecolor="red", bins=10)



# compare the values for both transactions
credit_card_data.groupby('Class').mean()



"""
Under-Sampling

Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

"""


legit_sample = legit.sample(492)



# Concatenating two DataFrames
new_dataset = legit_sample.append(fraud)



# Splitting the data into Features & Targets
X = new_dataset.iloc[:,:-1]
Y = new_dataset.iloc[:,-1]



#Split the data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)



# Model training using Logistic Regression
model = LogisticRegression()



# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)



# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)



# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)







