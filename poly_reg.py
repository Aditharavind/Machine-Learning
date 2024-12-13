import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset=pd.read_csv("poly.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

plt.scatter(x,y)
plt.plot(x_train,model.predict(x_train),color="red")
plt.title("Using Linear_reg")

#polynomial 

from sklearn.preprocessing import PolynomialFeatures
polynomial_reg=PolynomialFeatures(degree=6)#degree means no of limited polynomial
x_poly=polynomial_reg.fit_transform(x)#converted x to x_poly

linear_reg=LinearRegression()
linear_reg.fit(x_poly,y)


plt.scatter(x,y)
plt.plot(x,linear_reg.predict(x_poly),color="red")
plt.title("Using polynomial")

model.predict([[6.5]])
linear_reg.predict(polynomial_reg.fit_transform([[6.5]]))