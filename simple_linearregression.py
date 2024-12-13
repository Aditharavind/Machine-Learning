#importing lib
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_set=pd.read_csv("w.csv") # importing the dataset

#x=data_set.iloc[:,:-1].values # we import all the colum expect the last_one 
#y=data_set.iloc[:,-1].values  #we import the last col only 

#o
x=data_set[['years_experience']] 
y=data_set[['salary']]

#print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,model.predict(x_train),color="blue")

