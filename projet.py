# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 18:51:16 2024

@author: kerim
"""

import pickle
import numpy as np
import streamlit as st

model =pickle.load(open('tr_model_svm.sav','rb'))

scaler=pickle.load(open('scaler_.pkl','rb'))

def prediction(input_data):
    data_tab=np.asarray(input_data)
    data_r=data_tab.reshape(1,-1)
    x=scaler.transform(data_r)
    prediction=model.predict(x)
    if(prediction[0] ==0 ):
        return "Patient's not likely to have stroke"
    else :
        return "Patient's likely to have stroke"
    

def main ():
    st.title ('Patients Stroke Prediction')
    
 
    
    #gender	age	hypertension	heart_disease	avg_glucose_level	bmi	smoking_status_Unknown	smoking_status_formerly smoked	smoking_status_never smoked	smoking_status_smokes
    
    gender= st.radio('Patient s gender',['Male','Female'], index=None)
    if gender == "Male":
        result_g=1
    else :
        result_g=0
        
        
    age= st.number_input('Patient s age',min_value=0,value=0, step=1)
    hypertension =st.radio('Does the Patient have hypertension',['Yes','No'], index=None)
    if hypertension == "Yes":
        result_h=1
    else :
        result_h=0
    heart_disease= st.radio('Does the Patient have a heart desease',['Yes','No'], index=None)
    if heart_disease == "Yes":
        result_Hd=1
    else :
        result_Hd=0
    avg_glucose_level= st.number_input('Patient s average glucose level',min_value=0,value=0, step=1)
    bmi= st.number_input('Patient s  BMI',min_value=0,value=0, step=1)
    options = ["Unknown", "Formely smoked", "Never smoked", "Currently smoking"]
    smoking = st.radio("Patient s Smoking situation:",options, index=None )
    values = {option: 1 if option == smoking else 0 for option in options}
    
    sm_1 = 1 if smoking == "Unknown" else 0
    sm_2 = 1 if smoking == "Formely smoked" else 0
    sm_3 = 1 if smoking == "Never smoked" else 0
    sm_4 = 1 if smoking == "Currently smoking" else 0
    
    diagnosis=''
    
    
    if st.button('Result'):
        diagnosis=prediction([result_g,age,result_h,result_Hd,avg_glucose_level,bmi,sm_1,sm_2,sm_3,sm_4])
    
        
    st.success(diagnosis)
    
if __name__ == '__main__' :
    main( )
    




