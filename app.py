from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np


app=Flask(__name__)

with open("model.pkl",'rb') as model_file:
    model = pickle.load(open('model.pkl', 'rb'))

with open("scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    
    gender=request.form['gender']
    married=request.form['married']
    dependents=request.form['dependents']
    education=request.form['education']
    selfemployed=request.form['selfemployed']
    applincome=float(request.form['applincome'])
    coapplincome=float(request.form['coapplincome'])
    loanamount=float(request.form['loanamount'])
    loanamountterm=float(request.form['loanamountterm'])
    credithistory=float(request.form['credithistory'])
    propertyarea=request.form['propertyarea']
    

    new_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents':[dependents],
        'Education': [education],
        'Self_Employed': [selfemployed],
        'ApplicantIncome': [applincome],
        'CoapplicantIncome': [coapplincome],
        'LoanAmount': [loanamount],
        'Loan_Amount_Term': [loanamountterm],
        'Credit_History': [credithistory],
        'Property_Area': [propertyarea]
    })

    # Preprocessing new data (following the same steps as the training data)
    new_data['Dependents']=new_data['Dependents'].replace({'3+':4})
    new_data['Dependents']=new_data['Dependents'].astype('int64')
    new_data['Gender'] = new_data['Gender'].map({'Male': 1, 'Female': 0})
    new_data['Married'] = new_data['Married'].map({'Yes': 1, 'No': 0})
    new_data['Education'] = new_data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    new_data['Self_Employed'] = new_data['Self_Employed'].map({'Yes': 1, 'No': 0})

    # One-hot encode the 'Property_Area' column
    new_data = pd.get_dummies(new_data, columns=['Property_Area'], dtype='int64')

    # Ensure the new data has the same columns as the training data (after one-hot encoding)
    columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 
       'Property_Area_Rural', 'Property_Area_Semiurban',
       'Property_Area_Urban']
    missing_cols = set(columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    
    
    # Ensure the order of columns matches the training data
    
    new_data = new_data[columns]

    
    # Scale the numerical features
    columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    new_data[columns_to_scale] = scaler.transform(new_data[columns_to_scale])

    # Predict the outcome
    prediction = model.predict(new_data)
    prediction_result = 'Approved' if prediction == 1 else 'Not Approved'
    return render_template("predict.html",prediction_result=prediction_result)

if __name__=="__main__":
    app.run(debug=True)