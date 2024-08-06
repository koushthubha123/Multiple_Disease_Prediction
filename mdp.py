# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:10:52 2024

@author: Venkathimalay
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")




# loading the saved models

diabetes_model = pickle.load(open('C:/Users/Venkathimalay/OneDrive/Documents/MDP/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('C:/Users/Venkathimalay/OneDrive/Documents/MDP/saved_models/heart_disease_model.sav', 'rb'))

breast_cancer_model = pickle.load(open('C:/Users/Venkathimalay/OneDrive/Documents/MDP/saved_models/breast_cancer.sav', 'rb'))
liver_disease_model = pickle.load(open('C:/Users/Venkathimalay/OneDrive/Documents/MDP/saved_models/liver_disease_model.sav', 'rb'))

# Create the sidebar with the option menu
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Breast Cancer', 'Liver Disease'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person-standing-dress','activity'],
                           default_index=0) 

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies',min_value=0,max_value=100,format="%d")

    with col2:
        Glucose = st.number_input('Glucose Level')

    with col3:
        BloodPressure = st.number_input('Blood Pressure value')

    with col1:
        SkinThickness = st.number_input('Skin Thickness value')

    with col2:
        Insulin = st.number_input('Insulin Level')

    with col3:
        BMI = st.number_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value',format="%.3f")

    with col2:
        Age = st.number_input('Age',  min_value=0,max_value=150,format="%d")


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic 😷'
        else:
            diab_diagnosis = 'The person is not diabetic 😁'
            st.balloons()

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0,max_value=150,format="%d")

    with col2:
        sex = st.number_input('Sex (Male = 1,Female = 0)', min_value=0,max_value=1)

    with col3:
        cp = st.number_input('Chest Pain types')

    with col1:
        trestbps = st.number_input('Resting Blood Pressure')

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl',min_value=0,max_value=1,format="%d")

    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.number_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')

    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', min_value=0,max_value=2,format="%d")

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease 😷'
        else:
            heart_diagnosis = 'The person does not have any heart disease 😁'
            st.balloons()

    st.success(heart_diagnosis)

# Breast Cancer Prediction Page
if selected == 'Breast Cancer':

    # page title
    st.title('Breast Cancer Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        clump_thickness = st.number_input('clump_thickness', min_value=0.00, max_value=100.00, step=0.01, format="%.2f")
    with col2:
        unif_cell_size = st.number_input('unif_cell_size', min_value=0, max_value=100)

    with col3:
        unif_cell_shape = st.number_input('unif_cell_shape', min_value=0, max_value=100)

    with col1:
        marg_adhesion = st.number_input('marg_adhesion', min_value=0, max_value=100)

    with col2:
        single_epith_cell_size = st.number_input('single_epith_cell_size', min_value=0, max_value=100)

    with col3:
        bare_nuclei = st.number_input('bare_nuclei', min_value=0, max_value=100)

    with col1:
        bland_chrom = st.number_input('bland_chrom', min_value=0, max_value=100)

    with col2:
        norm_nucleoli = st.number_input('norm_nucleoli', min_value=0, max_value=100)

    with col3:
        mitoses = st.number_input('mitoses', min_value=0, max_value=100)

    # code for Prediction
    breast_cancer_diagnosis = ''

    # creating a button for Prediction

    if st.button('Breast Cancer Test Result'):

        user_input = [clump_thickness , unif_cell_size, unif_cell_shape, marg_adhesion, single_epith_cell_size, bare_nuclei, bland_chrom, norm_nucleoli, mitoses]

        user_input = [float(x) for x in user_input]

        breast_cancer_prediction = breast_cancer_model.predict([user_input])

        if breast_cancer_prediction[0] == 1:
            breast_cancer_diagnosis = 'The person is having Breast Cancer 😷'
        else:
            breast_cancer_diagnosis = 'The person does not have any Breast Cancer 😁'
            st.balloons()

    st.success(breast_cancer_diagnosis)

if selected == 'Liver Disease':

    # page title
    st.title('Liver Disease Prediction using ML')

    # st.markdown("<h1 style='font-style:italic;'>Liver Disease Prediction using ML</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input('Age',  min_value=0,max_value=150,format="%d")
    with col2:
        Gender = st.number_input('Gender (Male = 1,Female = 0)', min_value=0,max_value=1,format="%d")

    with col3:
        Total_Bilirubin = st.number_input('Total Bilirubin')

    with col1:
        Direct_Bilirubin = st.number_input('Direct Bilirubin')

    with col2:
        Alkaline_Phosphotase = st.number_input('Alkaline Phosphotase')

    with col3:
        Alamine_Aminotransferase = st.number_input('Alamine Aminotransferase')

    with col1:
        Aspartate_Aminotransferase = st.number_input('Aspartate Aminotransferase')

    with col2:
        Total_Protiens = st.number_input('Total Protiens')

    with col3:
        Albumin = st.number_input('Albumin')

    with col1:
        Albumin_and_Globulin_Ratio = st.number_input('Albumin and Globulin Ratio')

    # code for Prediction
    liver_diagnosis = ''

    
    if st.button('Liver Disease Test Result'):
        
        user_input = [Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,Albumin, Albumin_and_Globulin_Ratio]

        # Convert inputs to float
        user_input = [float(x) for x in user_input]

        # Make prediction using the trained model
        liver_prediction = liver_disease_model.predict([user_input])
        
        # Determine the diagnosis based on prediction
        if liver_prediction[0] == 1:
            liver_diagnosis = 'The person is having Liver disease 😷'
        else:
            liver_diagnosis = 'The person does not have any liver disease 😁'
            st.balloons()

    st.success(liver_diagnosis)