import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the data
breast_data = pd.read_csv('cancer.csv')

# Replace '?' with NaN
breast_data.replace('?', np.nan, inplace=True)

# Drop rows with NaN values
breast_data.dropna(inplace=True)

# Convert all columns to numeric
breast_data = breast_data.apply(pd.to_numeric, errors='coerce')

# Prepare the feature matrix and target vector
X = breast_data.drop(columns=['classes', 'id'])
Y = breast_data['classes']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

input_data = (8,10,10,8,7,10,9,7,1)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
    print('The Person does not have Breast Cancer')
else:
    print('The Person has Breast Cancer')


import pickle
import os
filename = 'breast_cancer.sav'
# pickle.dump(model, open(filename, 'wb'))
# # loading the saved model
# loaded_model = pickle.load(open('breast_cancer.sav', 'rb'))

save_directory = 'C:/Users/Venkathimalay/OneDrive/Documents/MDP/saved_models'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
print(save_directory)
# Full path to the file
filename = os.path.join(save_directory, filename)

# Save the trained model to the specified directory
pickle.dump(model, open(filename, 'wb'))

