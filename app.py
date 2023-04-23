import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('deployment.sav', 'rb'))


st.title('Loan Prediction App')

credit_policy = st.slider("Credit Policy",0,2,1)
int_rate = st.slider("Interest Rate",0.01,0.3,0.01)
installment = st.slider("Installment",10,1000,100)
log_annual_inc = st.slider("Log Annual Income",0,6,0.8)
dti = st.slider("Debt-to-Income Ratio",0,30,10)
fico = st.slider('FICO Credit Score',600,900,100)
days_with_cr_line = st.slider("Days with Credit Line",0,30000,1000)
revol_bal = st.slider("Revolving Balance",0,36000,1000)
revol_util = st.slider("Revolving Line Utilization Rate",0,101,1)
inq_last_6mths = st.slider("Inquiries in Last 6 Months",0,7,1)
delinq_2yrs = st.slider("Delinquencies in Last 2 Years",0,14,1)
pub_rec = st.slider("Public Records",0,6,1)
purpose_credit_card = st.slider('Purpose: Credit Card', 0,2,1)
purpose_debt_consolidation = st.slider('Purpose: Debt Consolidation',0,2,1)
purpose_educational = st.slider('Purpose: Educational',0,2,1)
purpose_home_improvement = st.slider('Purpose: Home Improvement', 0,2,1)
purpose_major_purchase = st.slider('Purpose: Major Purchase', 0,2,1)
purpose_small_business = st.slider('Purpose: Small Business',0,2,1)

def predict():
    features =[int(x) for x in [credit_policy, int_rate, installment, log_annual_inc, dti, fico, days_with_cr_line, revol_bal, revol_util,inq_last_6mths, delinq_2yrs, pub_rec, purpose_credit_card, purpose_debt_consolidation, purpose_educational,purpose_home_improvement, purpose_major_purchase, purpose_small_business]]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    if prediction == 1:
        st.success('The loan application is likely to be approved. :thumbsup:')
    else:
        st.warning('The loan application is likely to be rejected. :thumbsdown:')
trigger = st.button('Predict', on_click=predict)
