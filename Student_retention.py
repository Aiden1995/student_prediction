# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd

#url="https://app.box.com/s/yxm1vhhg9fnnpeflnaanil4znvb47n6b"
dataset = pd.read_csv('Student_retention.csv',sep=',') 
X = dataset.iloc[:,1:6]   #Independent variable
y = dataset.iloc[:, 6]      #Dependent variabl

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 4] = labelencoder_X.fit_transform(X.iloc[:, 4])  #Change the column value accordingly
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 1.Logistic Regression

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
                    


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
test1= [[0,2,2,1,0]]
y_pred1 = classifier.predict(test1)
confidence = classifier.predict_proba(test1)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Saving the model
from sklearn.externals import joblib
joblib.dump(classifier,'classification_model.pkl')
print('Model Dumped!!!!!!')


joblib.load('classification_model.pkl')

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")






























