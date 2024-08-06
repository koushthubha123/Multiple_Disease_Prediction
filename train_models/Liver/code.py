import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# loading the csv data to a Pandas DataFrame
liver_data = pd.read_csv('liver.csv')
# Mapping the Gender for Male =1 and Female = 0
liver_data['Gender'] = liver_data['Gender'].map({'Male': 1, 'Female': 0})

# getting some info about the data
liver_data.info()
# checking for missing values
liver_data.isnull().sum()
# adding 0 to the missing values
liver_data=liver_data.fillna(0)
liver_data.isnull().sum()


# statistical measures about the data
liver_data.describe()
X = liver_data.drop(columns=['Dataset'])
# Get the target column 'Dataset'
Y = liver_data['Dataset']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create and train the model
modelLiver = LogisticRegression()
modelLiver.fit(X_train, Y_train)

# Make predictions
Y_pred = modelLiver.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

input_data = (17,1,0.9,0.3,202,22,19,7.4,4.1,1.2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = modelLiver.predict(input_data_reshaped)
print(prediction)



if (prediction[0] == 2):
  print('The person does not have any liver disease')
elif(prediction[0] == 1):
  print('The person is having Liver disease')


input_data = (55,1,0.7,0.2,290,53,58,6.8,3.4,1)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = modelLiver.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 2):
  print('The person does not have any liver disease')
elif(prediction[0] == 1):
  print('The person is having Liver disease')



import pickle
import os
filename = 'liver_disease_model.sav'
save_directory = 'C:/Users/Venkathimalay/OneDrive/Documents/MDP/saved_models'
if not os.path.exists(save_directory):
 os.makedirs(save_directory)
print(save_directory)
# Full path to the file
filename = os.path.join(save_directory, filename)
# Save the trained model to the specified directory
pickle.dump(modelLiver, open(filename, 'wb'))


