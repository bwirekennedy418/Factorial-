import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("C:\pydatafiles\diabetes.csv") 
print(train.head())
print(train.info())
print(train.describe())
print(train.columns)
print(train.shape)
print(train.isnull().sum())

Encoder=preprocessing.LabelEncoder()
Encoded_train=train.apply(preprocessing.LabelEncoder().fit_transform)
print("Transformed Data:\n",Encoded_train)
Numeric_Array=Encoded_train.values
print("Numeric Array\n",Numeric_Array)

Training_Sample,Test_Sample=train_test_split(Numeric_Array,test_size=0.2,random_state=2)
print("Training Sample:\n",Training_Sample)
print("Test Sample:\n",Test_Sample)

XTrain_Sample=Training_Sample[:,1]
print("Input Attributes of Training Sample\n",XTrain_Sample)
YTrain_Sample=Training_Sample[:,-1]
print("Output Attributes of Training Sample\n\n",YTrain_Sample)
XTest_Sample=Test_Sample[:,:-1]
print("Input Attributes of Test Sample\n",XTest_Sample)
Actual_YTest_Sample=Test_Sample[:,-1]
print("Actual Test Sample Classes\n\n",Actual_YTest_Sample)
XTrain_Sample.shape
pca=PCA(n_components=2)
XTrain_Sample=Training_Sample[:,0:-1]
pca.fit(XTrain_Sample)
Decomposed_XTrain_Sample=pca.transform(XTrain_Sample)
print("\nDecomposed Input Attributes\n",Decomposed_XTrain_Sample)

y=pd.DataFrame(train['Outcome'])
y=y.values.ravel()
x=train.drop(['Outcome'],axis=1)
print(x)
print(y)
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=1255,test_size=0.25)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(train_x, train_y)
from sklearn.metrics import accuracy_score
pred_cv = model.predict(test_x)
accuracy= accuracy_score(test_y, pred_cv)
print(accuracy)
pred_train = model.predict(train_x)
validation= accuracy_score(train_y, pred_train)
print(validation)

#Saving the model
import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()


import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Pregnancies,Glucose , BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):   
 
   
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Pregnancies,Glucose , BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
     
    if prediction == 0:
        pred = ('f''You have a chance of having diabetes.\nProbability of you being a diabetic is {output_print}.\nEat clean and exercise regularly')
    else:
        pred = ('f''Congratulations, you are safe.\n Probability of you being a diabetic is {output_print}')
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Diabeteses Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Pregnancies = st.number_input("Total No of Pregnancies")
    Glucose = st.number_input("Glucose Level")
    BloodPressure = st.number_input("Measured Blood Pressure")
    SkinThickness = st.number_input("Thickness of your Skin")
    Insulin= st.number_input("Dosage of Insulin")
    BMI = st.number_input("Your Body Mass Index")
    DiabetesPedigreeFunction= st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("What is your Age to be Precise")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Pregnancies,Glucose , BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age) 
        st.success('Apolite Reminder is {}'.format(result))
        print(Outcome)
     
if __name__=='__main__': 
    main()
