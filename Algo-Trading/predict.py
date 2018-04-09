import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import numpy as np
df=pd.read_csv('data10.csv')
X=np.array(df.drop(['high'],1))
y=np.array(df['high'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf=LinearRegression()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)
prediction=clf.predict([20,3,2018,2820])

print(prediction)
accuracy2=clf.score(X_train,y_train)
#print(accuracy2)