#!/usr/bin/env python
# coding: utf-8

# In[173]:


import os
import glob
import pandas as pd
import numpy as np

home_dir = '/home/edinmsa/results1'
os.chdir(home_dir)
file_extension = '.csv'


all_filenames = [i for i in glob.glob(f"*{file_extension}")]

all_series = []
for index in range(len(all_filenames)):
    if index == 0:
        all_series.append(pd.read_csv(all_filenames[index])['Unnamed: 0'])
    all_series.append(pd.read_csv(all_filenames[index])['0'])

    
file_name = 'cc' 
pd.DataFrame(all_series).to_csv(f'/home/edinmsa/results1/{file_name}.csv', header=None, index=False)


