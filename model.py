import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv("train_ctrUa4K.csv")

for i in ['LoanAmount','Loan_Amount_Term']:
    df[i].fillna(df[i].median(),inplace=True)

for i in ['Gender','Married','Dependents','Self_Employed','Credit_History']:
    df[i].fillna(df[i].mode()[0],inplace=True)

df.drop(['Loan_ID'],axis=1,inplace=True)


df['Dependents']=df['Dependents'].replace({'3+':4})
df['Dependents']=df['Dependents'].astype('int64')

df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
df['Married'] = df['Married'].map({'Yes':1,'No':0})
df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0})
df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})

df=pd.get_dummies(df,columns=['Property_Area'],dtype='int64')

from sklearn.preprocessing import StandardScaler
columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)





x=df.drop('Loan_Status',axis=1)
y=df['Loan_Status']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

# Under-sampling the majority class
under_sampler = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
X_res, y_res = under_sampler.fit_resample(x_train, y_train)

# Over-sampling the minority class using SMOTE
smote = SMOTETomek(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_res, y_res)



from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', C=0.615848211066026)
lr.fit(x_train, y_train)

#pickle the trained model

with open('model.pkl','wb') as model_files:
  pickle.dump(lr,model_files)