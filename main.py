import streamlit as st
import numpy as np
import pandas as pd
data_ = pd.read_csv('encoder_train_csv')

st.title('SCA FINAL SUBMISSION')
st.write('# A Customer Segmentation Project')

Gender = st.selectbox('Gender', ('Male', 'Female'))
Ever_Married = st.selectbox('Have you ever been married?', ('Yes', 'No'))
Age = st.selectbox('How old are you?', data_['Age'].dropna().sort_values().unique())
Graduated = st.selectbox('Are you a graduate?', ('Yes', 'No'))
Profession = st.selectbox('Profession', data_['Profession'].dropna().unique())
Work_Experience = st.selectbox('Years of Work Experience', data_['Work_Experience'].dropna().sort_values().unique())
Spending_Score = st.selectbox('Spending Score', data_['Spending_Score'].dropna().sort_values().unique())
Family_Size = st.selectbox('Family Size', data_['Family_Size'].dropna().sort_values().unique())
Var_1 = st.selectbox('Category', data_['Var_1'].dropna().sort_values().unique())

import joblib
encoder = joblib.load('Encoded.pkl')
x = encoder.transform(np.array([Gender, Ever_Married, Age, Graduated, Profession, Work_Experience, Spending_Score, Family_Size, Var_1]).reshape(1,-1))


import lightgbm
model = lightgbm.Booster(model_file= 'Model_LGBM.txt')
if st.button('Predict'):
    y=model.predict(x)
    Y = np.argmax(y)
    if Y == 0:
        text = 'Category A'
    elif Y == 1:
        text = 'Category B'
    elif Y == 2:
        text = 'Category C'
    elif Y == 3:
        text = 'Category D'
    st.write(text)