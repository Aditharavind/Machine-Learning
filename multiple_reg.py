import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data_set=pd.read_csv("covid.csv")
x=data_set.iloc[:,:-1]
y=data_set.iloc[:,-1]

#so we need to encode the values in body temp to 0,1 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[1])],remainder = "passthrough")
x=np.array(ct.fit_transform(x))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
model=LinearRegression()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("r2_score",metrics.r2_score(y_pred,y_test))

#plotting 

plt.plot(y_pred)
plt.plot(y_train)
plt.show()

