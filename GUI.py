import PySimpleGUI as sg
import _tkinter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import pandas as pd
sg.theme('BluePurple')
from keras.layers import Dense
from script import *



layout = [
    [
        sg.Text('Enter values here')
    ],

    [
        sg.Text('Gender', size =(15, 1)),
        sg.Radio('M', 'group 1', key='-MALE-'),
        sg.Radio('F', 'group 1', key='-FEMALE-'),
    ],

    [
        sg.Text('Age', size =(15, 1)), sg.InputText(key='-AGE-')
    ],

    [
        sg.Text('Annual Salary', size =(15, 1)), sg.InputText(key='-ANNUAL_SALARY-')
    ],

    [
        sg.Text('Credit Card Debt', size =(15, 1)), sg.InputText(key='-CREDIT_DEBT-')
    ],

    [
        sg.Text('Net Worth', size =(15, 1)), sg.InputText(key='-NET_WORTH-')
    ],

    [
        sg.Button('OK'), sg.Button('Cancel')
    ] 

    ]

window = sg.Window('Car Purchase Predictor', layout)

while True:
    event, values = window.read()
    if event == "Cancel":
        break

    if event == "OK":
        if values["-MALE-"]==True:
            gender = 1
        elif values["-FEMALE-"]==True:
            gender = 0

        input_test_sample = np.array([[gender, values["-AGE-"], values["-ANNUAL_SALARY-"], values["-CREDIT_DEBT-"], values["-NET_WORTH-"]]])
        #Scale input test sample data
        input_test_sample_scaled = scaler_in.transform(input_test_sample)
        #Predict output
        output_predict_sample_scaled = model.predict(input_test_sample_scaled)
        #Print predicted output
        print('Predicted Output (Scaled) =', output_predict_sample_scaled)
        #Unscale output
        output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
        print('Predicted Output / Purchase Amount ', output_predict_sample)

        sg.popup('Predicted Amount/Purchase Amount' , output_predict_sample)

    elif event == sg.WINDOW_CLOSED:
        break


window.close()