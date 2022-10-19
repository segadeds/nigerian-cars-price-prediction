import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

pickle_in = open('model.bin', 'rb') 
model = pickle.load(pickle_in)

pickle_pr = open('preprocess.bin', 'rb') 
preprocess = pickle.load(pickle_pr)


@st.cache()

def prediction(Year_of_manufacture, Condition, Mileage, Engine_Size, Fuel, Transmission, Build):
     
    data = {'Year_of_manufacture': Year_of_manufacture,
        'Condition': Condition,
        'Mileage':Mileage,
        'Engine_Size':Engine_Size,
        'Fuel':Fuel,
        'Transmission':Transmission,
        'Build':Build
        }

    df = pd.DataFrame(data,index=[0])
    
    transform = preprocess.transform(df)
    
    
    prediction = model.predict(transform)
     
    prediction = np.exp(prediction)
    return prediction


def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:pink;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Nigerian Car Price Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Year_of_manufacture = st.number_input("Year of car manufacture")
    Condition = st.selectbox('Condition',("Nigerian Used","Foreign Used"))
    Mileage = st.number_input("Mileage")
    Engine_Size = st.number_input("Engine Size")
    Fuel = st.selectbox('Fuel',("Petrol","Hybrid","Diesel"))
    Transmission = st.selectbox('Transmission',("Manual","Automatic","CVT","AMT"))
    Build = st.selectbox('Build',("SUV","other"))
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Year_of_manufacture, Condition, Mileage, Engine_Size, Fuel, Transmission, Build) 
        result = result[[0][0]]
        st.success('The value of the vehicle is around ₦{}'.format(f'{result:,}'))
        
if __name__=='__main__': 
    main()
