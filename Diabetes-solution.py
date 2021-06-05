#!/usr/bin/env python
# coding: utf-8

# DIABETES DETECTION CHALLENGE

# In[30]:


#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[31]:


#Importing the dataset
dataset = pd.read_csv(r"C:\Users\CHARAN R\Desktop\Projects\Diabetes detection\diabetes.csv")
dataset.head()


# In[32]:


#Cleaning the Data

col = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for column in col:
    dataset[column] = dataset[column].replace(0,np.nan)    
    mean = int(dataset[column].mean(skipna=True))       
    dataset[column] = dataset[column].replace(np.nan,mean) 
                                                               


# In[33]:


#Spliting the Dataset

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1:]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)



# In[34]:


#Feauture Scaling
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# In[35]:


#KNN- MODEL

knn = KNeighborsClassifier(n_neighbors=11,p=2,metric = 'euclidean')
knn.fit(X_train,y_train.values.ravel())

#Predicting the values of Testing data
predict = knn.predict(X_test)


# In[36]:


#Evaluating the Model
cm = confusion_matrix(y_test,predict)
print(cm)


# In[37]:


print('The F1 score is : ' + str(f1_score(y_test,predict)))
print('The Accuracy is : ' + str(accuracy_score(y_test,predict)))


# In[ ]:




