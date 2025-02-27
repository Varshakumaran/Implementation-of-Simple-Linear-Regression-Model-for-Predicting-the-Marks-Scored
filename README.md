# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: varsha k
RegisterNumber:  212223220122

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,  mean_squared_error
df=pd.read_csv('student_scores.csv')

df.head()


X = df.iloc[:,:-1].values
X

Y = df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

mse=mean_squared_error(Y_test,Y_pred)
print('MSE =',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE =',mae)

rmse=np.sqrt(mse)
print("RMSE =",rmse)



plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,Y_pred,color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

*/
```

## Output:
![Screenshot 2025-02-27 180947](https://github.com/user-attachments/assets/1c95a4c2-cfe0-42e8-a217-fc63d84041f8)

![Screenshot 2025-02-27 180942](https://github.com/user-attachments/assets/618912ba-208a-44cd-a1ed-07c6ae1d3122)

![Screenshot 2025-02-27 180935](https://github.com/user-attachments/assets/e5f6becc-a039-4b88-8104-4c90cc975993)

![Screenshot 2025-02-27 180927](https://github.com/user-attachments/assets/11fa7491-65ec-41f8-9629-839506f93bda)

![Screenshot 2025-02-27 180921](https://github.com/user-attachments/assets/086866ac-f0f2-4016-a14f-b6e9c6d6919a)

![Screenshot 2025-02-24 162129](https://github.com/user-attachments/assets/0644f664-9747-4c6a-baf5-0d6a6b109c79)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
