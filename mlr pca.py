import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importing the data
df = pd.read_csv('Admission_Predict_Ver1.1.csv')


#Extracting x and y values
x=df.iloc[:,1:8].values
y=df.iloc[:,-1].values

#standardising the data
stsc=StandardScaler()
x=stsc.fit_transform(x)

pca = PCA(n_components=1)
x = pca.fit_transform(x)
var = pca.explained_variance_ratio_
print(var)

x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=0)

#Creating and training the linear regression model
lr=LinearRegression()
lr.fit(x_tr,y_tr)

#Predicting the values with trained model
y_pred=lr.predict(x_te)

plt.scatter(x_te,y_te,c='blue')   
plt.plot(x_te,y_pred,c='grey')
plt.show()
