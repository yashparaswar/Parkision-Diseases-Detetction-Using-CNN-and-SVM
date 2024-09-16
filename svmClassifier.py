#Step1: Loading the required packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#Step2: Loading the dataset
data_set = pd.read_csv('events.csv')
data_set.dropna(inplace=True)
data_set.to_csv('cleanedEvents.csv',index=False)
print(data_set.head())



#Step3: Extract predictors & Target varoables
x = data_set.iloc[:,[1,2,3]].values
y = data_set.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_x = LabelEncoder()
#print("Before Encoding:",x[:,3])
x[:,2] = labelencoder_x.fit_transform(x[:,2].tolist())

print(x)


#Step4: splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Step5: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#Step6: Fitting the SVM classifier
from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)

#Step7: Performance
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

#acc = (TP+TN) / (TP+TN+FP+FN)   = (54+21) /(54+21+4+1)
'''
#Step8: Prediction
age = 45
sal = 20000
test_sample = [[age,sal]]
test_sample = sc.fit_transform(test_sample)

print("Result:",classifier.predict(test_sample))
'''

def predsvm(test_sample):
    test_sample = sc.transform(test_sample)
    return classifier.predict(test_sample)