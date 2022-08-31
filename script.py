import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import PySimpleGUI as sg
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense


#Data Import
data = pd.read_csv('car_sales_dataset.csv', encoding = 'ISO-8859-1')
print(data)

#Plotting Data
sns.pairplot(data)
plt.show()

#Create input dataset from data
inputs = data.drop(['Customer_Name', 'Customer_Email', 'Country', 'Purchase_Amount'], axis = 1)
#Show Input Data
print(inputs)
#Show Input Shape
print("Input data Shape=",inputs.shape)


#Create output dataset from data
output = data['Purchase_Amount']
#Show Output Data
print(output)
#Transform Output
output = output.values.reshape(-1,1)
#Show Output Transformed Shape
print("Output Data Shape=",output.shape)

#Scale input
scaler_in = MinMaxScaler()
input_scaled = scaler_in.fit_transform(inputs)
print(input_scaled)

#Scale output
scaler_out = MinMaxScaler()
output_scaled = scaler_out.fit_transform(output)
print(output_scaled)

#Create model
model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
print(model.summary())

#Train model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
epochs_hist = model.fit(input_scaled, output_scaled, epochs=20, batch_size=10, verbose=1, validation_split=0.2)
print(epochs_hist.history.keys()) #print dictionary keys

#Plot the training graph to see how quickly the model learns
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()


input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])

#Scale input test sample data
input_test_sample_scaled = scaler_in.transform(input_test_sample)

#Predict output
output_predict_sample_scaled = model.predict(input_test_sample_scaled)

#Print predicted output
print('Predicted Output (Scaled) =', output_predict_sample_scaled)

#Unscale output
output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
print('Predicted Output / Purchase Amount ', output_predict_sample)
