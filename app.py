import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import base64

st.sidebar.title('Transaction Information')

html_temp = """
<div style="background-color:Blue;padding:10px">
<h2 style="color:white;text-align:center;">Fraud Detection</h2>
<h4 style="color:black;text-align:center;">Team Data Fishers</h4>
</div><br>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: Black;'>Select Your Model</h1>", unsafe_allow_html=True)

selection = st.selectbox("", ["Logistic Regression", "Random Forest", "XGBClassifier"])

if selection == "Logistic Regression":
    st.write("You selected", selection, "model")
    model = pickle.load(open('mlr_model.pkl', 'rb'))
elif selection == "Random Forest":
    st.write("You selected", selection, "model")
    model = pickle.load(open('mrf_model.pkl', 'rb'))
else:
    st.write("You selected", selection, "model")
    model = pickle.load(open('lxgb_model.pkl', 'rb'))

Amount = st.sidebar.slider(label="Amount", min_value=10000000.00, max_value=100.00, step=0.01)
CreditLimit  = st.sidebar.slider(label="Credit Limit", min_value=10000000.00, max_value=100.00, step=0.01)
Marital_Status = st.sidebar.slider(label="Marital Status", min_value=10000000.00, max_value=100.00, step=0.01)
Gender = st.sidebar.slider(label="Gender", min_value=10000000.00, max_value=100.00, step=0.01)
Domain = st.sidebar.slider(label="Domain", min_value=10000000.00, max_value=100.00, step=0.01)
AverageIncomeExpendicture = st.sidebar.slider(label="Average Income Expendicture", min_value=10000000.00, max_value=100.00, step=0.01)
Cards  = st.sidebar.slider(label="Cards", min_value=10000000.00, max_value=100.00, step=0.01)
NewBalance = st.sidebar.slider(label="New Balance", min_value=10000000.00, max_value=100.00, step=0.01)
TransactionType = st.sidebar.slider(label="Transaction Type", min_value=10000000.00, max_value=100.00, step=0.01)
CustomerAge = st.sidebar.slider(label="CustomerAge", min_value=10000000.00, max_value=100.00, step=0.01)
Amount1 = st.sidebar.slider(label="Amount1", min_value=10000000.00, max_value=100.00, step=0.01)
CreditLimit1  = st.sidebar.slider(label="Credit Limit1", min_value=10000000.00, max_value=100.00, step=0.01)
Marital_Status1 = st.sidebar.slider(label="Marital Status1", min_value=10000000.00, max_value=100.00, step=0.01)
Gender1 = st.sidebar.slider(label="Gender1", min_value=10000000.00, max_value=100.00, step=0.01)
Domain1 = st.sidebar.slider(label="Domain1", min_value=10000000.00, max_value=100.00, step=0.01)
Amount2 = st.sidebar.slider(label="Amount2", min_value=10000000.00, max_value=100.00, step=0.01)
CreditLimit2  = st.sidebar.slider(label="Credit Limit2", min_value=10000000.00, max_value=100.00, step=0.01)
Marital_Status2 = st.sidebar.slider(label="Marital Status2", min_value=10000000.00, max_value=100.00, step=0.01)
Gender2 = st.sidebar.slider(label="Gender2", min_value=10000000.00, max_value=100.00, step=0.01)
Domain2 = st.sidebar.slider(label="Domain2", min_value=10000000.00, max_value=100.00, step=0.01)
Amount3 = st.sidebar.slider(label="Amount3", min_value=10000000.00, max_value=100.00, step=0.01)
CreditLimit3  = st.sidebar.slider(label="Credit Limit3", min_value=10000000.00, max_value=100.00, step=0.01)
Marital_Status3 = st.sidebar.slider(label="Marital Status3", min_value=10000000.00, max_value=100.00, step=0.01)
Gender3 = st.sidebar.slider(label="Gender3", min_value=10000000.00, max_value=100.00, step=0.01)
Domain3 = st.sidebar.slider(label="Domain3", min_value=10000000.00, max_value=100.00, step=0.01)
Amount4 = st.sidebar.slider(label="Amount4", min_value=10000000.00, max_value=100.00, step=0.01)
CreditLimit4  = st.sidebar.slider(label="Credit Limit4", min_value=10000000.00, max_value=100.00, step=0.01)
Marital_Status4 = st.sidebar.slider(label="Marital Status4", min_value=10000000.00, max_value=100.00, step=0.01)
Gender4 = st.sidebar.slider(label="Gender4", min_value=10000000.00, max_value=100.00, step=0.01)
CreditLimit5 = st.sidebar.slider(label="Credit Limit5", min_value=10000000.00, max_value=100.00, step=0.01)
Marital_Status5 = st.sidebar.slider(label="Marital Status5", min_value=10000000.00, max_value=100.00, step=0.01)
Gender5 = st.sidebar.slider(label="Gender5", min_value=10000000.00, max_value=100.00, step=0.01)


col1_dict = {'Amount':Amount, 'Credit Limit':CreditLimit, 'Marital Status':Marital_Status, 'Gender':Gender, 'Domain':Domain,
             'Average Income Expendicture':AverageIncomeExpendicture, 'Cards':Cards, 'New Balance':NewBalance, 'Transaction Type':TransactionType, 'CustomerAge':CustomerAge,
             'Amount1':Amount1, 'Credit Limit1':CreditLimit1, 'Marital Status1':Marital_Status1, 'Gender1':Gender1, 'Domain':Domain1,
             'Amount2':Amount2, 'Credit Limit2':CreditLimit2, 'Marital Status2':Marital_Status2, 'Gender2':Gender2, 'Domain':Domain2,
             'Amount3':Amount3, 'Credit Limit3':CreditLimit3, 'Marital Status3':Marital_Status3, 'Gender3':Gender3, 'Domain':Domain3,
             'Amount4':Amount4, 'Credit Limit4':CreditLimit4, 'Marital Status4':Marital_Status4, 'Gender4':Gender4,
             'Credit Limit5':CreditLimit5, 'Marital Status5':Marital_Status5, 'Gender5':Gender5
             
            }

columns = ['Amount', 'CreditLimit', 'Martial_Status', 'Gender', 'Domain', 
           'AverageIncomeExpendicture', 'Cards', 'NewBalance', 'TransactionType', 'CustomerAge', 
           'Amount1', 'CreditLimit1', 'Martial_Status1', 'Gender1', 'Domain1', 
           'Amount2', 'CreditLimit2', 'Martial_Status2', 'Gender2', 'Domain2',
           'Amount3', 'CreditLimit3', 'Martial_Status3', 'Gender3', 'Domain3',
           'Amount4', 'CreditLimit4', 'Martial_Status4', 'Gender4', 'CreditLimit5', 'Martial_Status5', 'Gender5']

df_coll = pd.DataFrame.from_dict([col1_dict])
user_inputs = df_coll

prediction = model.predict(user_inputs)

html_temp = """
<div style="background-color:Black;padding:10px">
<h2 style="color:white;text-align:center;">Fraud Detection Prediction</h2>
</div><br>
"""
st.markdown("<h1 style='text-align: center; color: Black;'>Transaction Information</h1>", unsafe_allow_html=True)

st.table(df_coll)

st.subheader('Click PREDICT if configuration is OK')

if st.button('PREDICT'):
    if prediction[0]==0:
        st.success(prediction[0])
        st.success('Transaction is SAFE :)')
    elif prediction[0]==1:
        st.warning(prediction[0])
        st.warning('ALARM! Transaction is FRAUDULENT :(')
