from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.chdir('../csv_files/')

file_list = os.listdir('./')
for file in file_list:
    if '~' in file:
        file_list.remove(file)

def check_null(data):
    for col in data.columns:
        print (col,':',data[col].isnull().sum())
    print('\n')
#def pipeline(file):

null_list = [99.0,999.0,99,999,9999.0,9999]

for file in file_list:
    #pipeline(file)
    data = pd.read_csv(file,  skiprows=range(1, 2))
    drop_list = list()
    attrs = data.columns
    print('\n\n======================')
    print(file.split('.')[0])
    print('======================\n')
    #Printing nullsums and assigning np.nan
    for attr in attrs:
        null_sum = 0
        m = data[attr].max()
        if m in null_list: 
            null_sum = (data[attr] == m).sum()
            data[attr] = data[attr].replace(m, np.nan)
        print (attr,' : ', null_sum)
        if null_sum > 0.2 * data.shape[0]:
            drop_list.append(attr)
    print('\nPrinting Drop List :')
    print (drop_list)
    data = data.drop(drop_list, axis=1)
    print('\nNew Data Columns :')
    print (data.columns)
    attrs = data.columns[:5]
    for attr in  attrs:
        #print (attr)    
        data[attr] = data[attr].astype(str)
        data[attr] = data[attr].str.zfill(2)
        #print (data[attr])
    data['timestamp'] = data['#YY'] + '/' + data['MM'] + '/' + data['DD'] + ':' + data['hh'] + ':' + data['mm']
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y/%m/%d:%H:%M')
    ind = pd.Index(data['timestamp'])
    data.index = ind
    data = data.drop(['#YY', 'MM','DD', 'hh', 'mm', 'timestamp'], axis = 1)
    """cols = data.columns
    print('\nPrinting Max of Each COL')
    for col in cols:
        max = data[col].max()
        print (col, max)
        data[col] = data[col].replace(max, np.nan)"""
    print('\n')
    """    data.WDIR = data.WDIR.replace(999,np.nan)
    data.WSPD = data.WSPD.replace(99.0, np.nan)
    data.GST = data.GST.replace(99.0, np.nan)
    data.PRES = data.PRES.replace(9999.0, np.nan)
    data.WTMP = data.WTMP.replace(999.0, np.nan)
    data.ATMP = data.ATMP.replace(999.0, np.nan)"""
    print('\nPrinting Null Sum Before')
    check_null(data)
    data = data.fillna(method="ffill")
    print('\nPrinting Null Sum After')
    check_null(data)
    #R_data = data.drop(['#YY', 'MM','DD', 'hh', 'mm', 'timestamp'], axis = 1)
    #print (data.head())
    data = data.resample('60T').mean()
    print ('\n',data.shape)
    #data = data.dropna(how = 'all')
    print('\nPrinting Null Sum Finally')
    check_null(data)
    content = data.to_csv()
    with open('../preprocessed_data/cleaned_' + file, 'w') as hand:
        hand.write(content)
