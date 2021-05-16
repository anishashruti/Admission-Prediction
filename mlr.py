import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importing the dataset
df = pd.read_csv('Admission_Predict_Ver1.1.csv')


#Extracting x and y values
x=df.iloc[:,1:8].values
y=df.iloc[:,-1].values

stsc=StandardScaler()
x=stsc.fit_transform(x)

#Splitting x and y into training set and test set
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=0)


#Creating and training the linear regression model
lr=LinearRegression()
lr.fit(x_tr,y_tr)

#Predicting the values with trained model
y_pred=lr.predict(x_te)

#Printing the r^2 score and errors of out trained model
print('Absolute error is: '+ str(mean_absolute_error(y_te,y_pred))) #0.048253893748654735
mse = mean_squared_error(y_te,y_pred)
print('Mean square error is: '+ str(mse))                           #0.0040796800346021055
print(r2_score(y_te,y_pred))                     #0.7664048993199384
print('Root mean square error is:'+ str(np.sqrt(mse))) #0.06387237301527246

x_copy=x[:,0:]
const=np.ones((500,1)).astype(int)
x_copy=np.append(arr=const,values=x_copy,axis=1)
x_opt=np.array(x_copy[:,[0,1,2,3,4,5,6,7]],dtype=float)
p_value=sm.OLS(endog=y,exog=x_opt).fit()
#print(p_value.summary())

x_opt=np.array(x_copy[:,[0,1,2,3,5,6,7]],dtype=float)
p_value=sm.OLS(endog=y,exog=x_opt).fit()
#print(p_value.summary())

x_opt=np.array(x_copy[:,[0,1,2,5,6,7]],dtype=float)
p_value=sm.OLS(endog=y,exog=x_opt).fit()
#print(p_value.summary())

x_train,x_test,y_train,y_test=train_test_split(x_opt,y,test_size=0.2,random_state=0)

opt_reg=LinearRegression()
opt_reg.fit(x_train,y_train)

y_pred=opt_reg.predict(x_test)

print(mean_absolute_error(y_test,y_pred))  #0.048253893748654735 ------#0.04885159906248055
mse = mean_squared_error(y_test,y_pred) 
print(mse) #0.0040796800346021055--------- #0.004126719133136814
print(r2_score(y_test,y_pred))  #0.7664048993199384------#0.76371152560804971
print(np.sqrt(mse)) #0.06387237301527246-------#0.06423954493251656

plt.scatter(x_test[:,4],y_test,c='BLue')  
plt.plot(x_test[:,4],y_pred,c='Gray')
plt.show()