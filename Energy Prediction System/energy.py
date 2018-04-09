import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import numpy as np
df=pd.read_csv('sqa_dataset.txt')
X=np.array(df.drop(['Energy','Observation'],1))
#X=np.array(df.drop(['Energy',],1))
#X=np.array(df.drop(['Observation',],1))

#print(X)
print (X.shape)
y=np.array(df['Energy'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf=RandomForestRegressor()
clf.fit(X_train,y_train)

accuracy=clf.score(X_train,y_train)
print(accuracy)


accuracy1=clf.score(X_test,y_test)
print(accuracy1)

