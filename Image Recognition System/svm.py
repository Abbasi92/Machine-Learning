import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


filename= 'catdog_datasets.txt'
filename1= 'catdog_datasets.txt'
raw_data = open(filename, 'rt')
raw_data1 = open(filename1, 'rt')

features_data = np.loadtxt(raw_data,dtype='object',delimiter=":",usecols=(0))
labels_data = np.loadtxt(raw_data1,dtype='object',delimiter=":",usecols=(1))

labels_data = labels_data.astype(np.float)
#print(labels_data)

features_data = np.array([l.strip().split(' ') for l in features_data],dtype =np.float)
#print(features_data)


X_train, X_test, y_train, y_test = train_test_split(features_data,labels_data,test_size=0.2)

#print (X_train.shape)
#print (y_train.shape)
#print (X_test.shape)
#print (y_test.shape)



clf = SVC(kernel='linear',C=1.5,gamma=0.10000000000000001)
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)
clf.decision_function(X_test)
print (predictions)
print (y_test)

accuracy = clf.score(X_test, y_test)
print (accuracy)
