#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import sklearn.naive_bayes


# In[19]:


training=pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_train.csv')
testing=pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_test.csv')
training


# In[24]:


from sklearn.naive_bayes import GaussianNB


# In[29]:


trainData = pd.DataFrame.values(training[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])
trainTarget = pd.DataFrame.values(training[['Species']]).ravel()
testData = pd.DataFrame.values(testing[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])
testTarget = pd.DataFrame.values(testing[['Species']]).ravel()


# In[55]:


trainData =pd.DataFrame(training[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]).values
trainTarget =pd.DataFrame(training[['Species']]).values.ravel()
testData = pd.DataFrame(testing[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]).values
testTarget = pd.DataFrame(testing[['Species']]).values.ravel()


# In[56]:


classifier = GaussianNB()
classifier.fit(trainData, trainTarget)


# In[58]:


predictedValues = classifier.predict(testData)

nErrors = (testTarget != predictedValues).sum()
accuracy = 1.0 - nErrors / testTarget.shape[0]
print("Accuracy: ", accuracy)


# In[63]:


results_nm = confusion_matrix(testTarget,predictedValues)
results_nm


# In[60]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import classification_report

