import pandas as pd
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])
#print (X)
#with open('out.txt','w') as out:
#	out.write(str(y))
	
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print(accuracy)

plt.title('Cancer')
plt.xlabel('Malignant')
plt.ylabel('Benign')
plt.legend()
plt.show()

#example_measures=np.array([4,2,1,1,1,2,3,2,1])
#print(example_measures)
#example_measures=example_measures.reshape(1,-1)
#print('*******')
#print(example_measures)
#prediction=clf.predict(example_measures)
#print(prediction)
#print(y_test)


