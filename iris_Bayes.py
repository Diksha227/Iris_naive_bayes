#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn.naive_bayes


# In[2]:


from sklearn.naive_bayes import GaussianNB


# In[3]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import classification_report


# In[4]:


training=pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_train.csv')
testing=pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_test.csv')
training


# In[5]:


trainData =pd.DataFrame(training[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]).values
trainTarget =pd.DataFrame(training[['Species']]).values.ravel()
testData = pd.DataFrame(testing[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]).values
testTarget = pd.DataFrame(testing[['Species']]).values.ravel()


# In[6]:


classifier = GaussianNB()
classifier.fit(trainData, trainTarget)


# In[7]:


predictedValues = classifier.predict(testData)

nErrors = (testTarget != predictedValues).sum()
accuracy = 1.0 - nErrors / testTarget.shape[0]
print("Accuracy: ", accuracy)


# In[8]:


results_nm = confusion_matrix(testTarget,predictedValues)
results_nm


# In[9]:


results = classification_report(testTarget,predictedValues)
results

