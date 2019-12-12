#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


iris=load_iris()


# In[4]:


features=iris.data.T


# In[5]:


sepal_length=features[0]
sepal_width=features[1]
petal_length=features[2]
petal_width=features[3]


# In[20]:


sepal_length_label=iris.feature_names[0]
sepal_width_label=iris.feature_names[1]
petal_length_label=iris.feature_names[2]
petal_width_label=iris.feature_names[3]


# In[21]:


plt.scatter(sepal_length,sepal_width,c=iris.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
plt.show()


# In[29]:


X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'],test_size=0.25,random_state=0)


# In[30]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[31]:


knn.fit(X_train,y_train)


# In[32]:


X_new=([[6.0,2.9,4.5,1.05]])


# In[33]:


prediction=knn.predict(X_new)


# In[34]:


prediction


# In[35]:


knn.score(X_test,y_test)


# In[ ]:




