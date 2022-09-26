#!/usr/bin/env python
# coding: utf-8

# 
# 

# In[ ]:


# put this file uder the forlder of task1b(contain "train.csv", "sample.csv")


# In[1]:


import pandas as pd
from sklearn import linear_model

import math


# In[2]:


path = "./train.csv"


# In[3]:


data = pd.read_csv(path)


# In[5]:


y = data["y"] # column 0 is id, column 1 is "y"
X_raw = data[data.columns[2:]] # 5 features from train.csv


# In[7]:


X=pd.DataFrame()


# In[8]:


col_id = []
for i in range(1,6):
    X["x"+str(i)] = X_raw.iloc[:,i-1]
    X["x"+str(i+5)] = X_raw.iloc[:,i-1].map(lambda x: x**2)
    X["x"+str(i+10)] = X_raw.iloc[:,i-1].map(lambda x: math.exp(x))
    X["x"+str(i+15)] = X_raw.iloc[:,i-1].map(lambda x: math.cos(x))
    


# In[9]:


X = X.iloc[:,[0,4,8,12,16,1,5,9,13,17,2,6,10,14,18,3,7,11,15,19]]


# In[12]:


# X.insert(X.shape[1], 'x21', 1)


# In[11]:


reg = linear_model.LinearRegression()
reg.fit(X, y)


# In[19]:


weights = reg.coef_ 


# In[21]:


# To csv
out_path = "./submission_task1b.csv"
result = pd.Series(weights)


# In[28]:


result[21] = reg.intercept_
# note: don't need to add "x21"=1! weight21 = intercept!


# In[30]:


result.to_csv(out_path, header=False, index=False)


# In[ ]:




