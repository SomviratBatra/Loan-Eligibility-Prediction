# Loan Eligibility Prediction Using ML
import pandas as pd

data = pd.read_csv('loan_prediction.csv')

#  COLUMN DETAILS

data.head()

data.tail()

data.shape

print("Number of Rows: ",data.shape[0])
print("Number of Columns: ",data.shape[1])

data.info()

data.isnull().sum()

len(data)

data.isnull().sum()*100 / len(data)

data = data.drop('Loan_ID' , axis=1)

data.head(2)

# making a list of columns with missing percentage < 5%

columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']

data = data.dropna(subset=columns)

data.isnull().sum()*100 / len(data)

# All columns, except **'Self_Employed'** and **'Credit_History'** are handled and these column's missing percentage is more than 5%, so we can't delete row them, we've to fill the missing values with appropriate values.
data['Self_Employed'].unique()

data['Credit_History'].unique()

data['Self_Employed'].mode()[0]

data['Credit_History'].mode()[0]

data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])

data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

data.isnull().sum()*100/len(data)

# - All Missing Values are Handled
data.sample(5)

data['Dependents'].unique()

#replace 3+ with 3
data['Dependents'] = data['Dependents'].replace(to_replace="3+",value='3')

data['Loan_Status'].unique()

#  Encoding

data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')

data.head()

import numpy as np
#import seaborn as sns

# Percentage of people of take loan by Gender

#print("number of people who take loan as group by gender:")
#print(data['Gender'].value_counts())

#sns.countplot(x='Gender', data = data, palette = 'Set1')

# Percentage of people of take loan by Marital Status

#print("number of people who take loan as group by marital status:")
#print(data['Married'].value_counts())

#sns.countplot(x='Married', data = data, palette = 'Set1')

# Percentage of people of take loan by Dependents

#print("number of people who take loan as group by dependents:")
#print(data['Dependents'].value_counts())

#sns.countplot(x='Dependents', data = data, palette = 'Set1')

# Percentage of people of take loan by Self_Employed

#print("number of people who take loan as group by Self Employed:")
#print(data['Self_Employed'].value_counts())

#sns.countplot(x='Self_Employed', data = data, palette = 'Set1')

# Percentage of people of take loan by Loan_Amount

#print("number of people who take loan as group by Loan Amount:")
#print(data['LoanAmount'].value_counts())

#sns.countplot(x='LoanAmount', data = data, palette = 'Set1')

# Percentage of people of take loan by Credit History

#print("number of people who take loan as group by Credit History:")
#print(data['Credit_History'].value_counts())

#sns.countplot(x='Credit_History', data = data, palette = 'Set1')

X = data.drop('Loan_Status', axis=1)

y = data['Loan_Status']

X

y

#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import accuracy_score

model_df = {}

def model_val(model,X,y):
    # spliting dataset for training and testing
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                   test_size=0.20,
                                                   random_state=42)
    
    # training the model
    model.fit(X_train, y_train)
    
    # asking model for prediction
    y_pred = model.predict(X_test)
    
    # checking model's prediction accuracy
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    
    # to find the best model we use cross-validation, thru this we can compare different algorithms
    # In this we use whole dataset to for testing not just 20%, but one at a time and summarize 
    # the result at the end.
    
    # 5-fold cross-validation (but 10-fold cross-validation is common in practise)
    score = cross_val_score(model,X,y,cv=5)  # it will divides the dataset into 5 parts and during each iteration 
                                             # uses (4,1) combination for training and testing 
    
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model] = round(np.mean(score)*100,2)
    

#  Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# passing this model object of LogisticRegression Class in the function we've created
model_val(model,X,y)

model_df

#  Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model_val(model,X,y)

model_df

#  SVC (Support Vector Classifier)
from sklearn import svm

model = svm.SVC()
model_val(model,X,y)

model_df
 
#  Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model =RandomForestClassifier()
model_val(model,X,y)

model_df

from sklearn.linear_model import LogisticRegression
import joblib


joblib.dump(data, 'trained_model.pkl')


df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':0,
    'Education':1,
    'Self_Employed':0,
    'ApplicantIncome':5720,
    'CoapplicantIncome':0,
    'LoanAmount':110,
    'Loan_Amount_Term':360,
    'Credit_History':1,
    'Property_Area':1
},index=[0])

result = model.predict(df)

if result==1:
    print("Loan Approved")
else:
    print("Loan Not Approved")

test = pd.read_csv('loan-test.csv')
test.head(3)

test.isnull().sum()

test.isnull().sum()*100/len(test)

cols = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']

test = test.dropna(subset=cols)

test['Self_Employed'] = test['Self_Employed'].fillna(test['Self_Employed'].mode()[0])

test['Credit_History'] = test['Credit_History'].fillna(test['Credit_History'].mode()[0])

test.isnull().sum()

test['Dependents'].unique()

test['Dependents'] = test['Dependents'].replace(to_replace="3+",value='3')

test = test.drop('Loan_ID' , axis=1)

test.head()

test['Gender'] = test['Gender'].map({'Male':1,'Female':0}).astype('int')
test['Married'] = test['Married'].map({'Yes':1,'No':0}).astype('int')
test['Education'] = test['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
test['Self_Employed'] = test['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
test['Property_Area'] = test['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')


rslt = model.predict(test)

rslt

