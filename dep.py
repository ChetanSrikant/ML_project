from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('loan_deployment.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    credit_policy = int(request.form['credit_policy'])
    int_rate = float(request.form['int_rate'])
    installment = float(request.form['installment'])
    log_annual_inc = float(request.form['log_annual_inc'])
    dti = float(request.form['dti'])
    fico = float(request.form['fico'])
    days_with_cr_line = float(request.form['days_with_cr_line'])
    revol_bal = float(request.form['revol_bal'])
    revol_util = float(request.form['revol_util'])
    inq_last_6mths = int(request.form['inq_last_6mths'])
    delinq_2yrs = int(request.form['delinq_2yrs'])
    pub_rec = int(request.form['pub_rec'])
    purpose_credit_card = int(request.form['purpose_credit_card'])
    purpose_debt_consolidation = int(request.form['purpose_debt_consolidation'])
    purpose_educational = int(request.form['purpose_educational'])
    purpose_home_improvement = int(request.form['purpose_home_improvement'])
    purpose_major_purchase = int(request.form['purpose_major_purchase'])
    purpose_small_business = int(request.form['purpose_small_business'])

    features = [credit_policy, int_rate, installment, log_annual_inc, dti, fico, days_with_cr_line, revol_bal, revol_util,
                inq_last_6mths, delinq_2yrs, pub_rec, purpose_credit_card, purpose_debt_consolidation,
                purpose_educational, purpose_home_improvement, purpose_major_purchase, purpose_small_business]

    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        result = 'Loan Rejected'
    else:
        result = 'Loan Approved'

    return render_template('index.html', prediction_text='{}'.format(result))
