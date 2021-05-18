#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests


# In[2]:


# ~~~~ load price 1-minute data and filter for hourly ~~~~


# In[3]:


dfbh = pd.read_csv('/home/ubuntu/notebooks/deeplearning/project/data/btc-masterdata.csv')


# In[4]:


val = dfbh['datetime'].loc[lambda x: x=='2019-04-01 00:00:00'].index.tolist()
dfb = dfbh[val[0]:]

dfb = dfb.reset_index()
dfb = dfb.drop(columns=['index'])


# In[5]:


dfb['datetime'] = dfb['datetime'].astype('datetime64')


# In[6]:


dfb = dfb[dfb.index % 60 == 0]


# In[7]:


# ~~~~ load feature data ~~~~


# In[8]:


df = pd.read_csv('/home/ubuntu/notebooks/deeplearning/project/data/datafile2.csv')


# In[9]:


df.tail()


# In[10]:


df.head()


# In[11]:


datadf = pd.DataFrame({'year':df['year'],'month':df['month'],'day':df['day']})


# In[12]:


df.insert(0, 'datetime', pd.to_datetime(datadf).dt.floor('D').dt.strftime('%Y-%m-%d %H:%M:%S'))


# In[13]:


df = df.drop(columns=['year', 'month', 'day'])


# In[14]:


df['datetime'] = df['datetime'].astype('datetime64')


# In[ ]:





# In[15]:


# ~~~~ merge datasets and interpolate ~~~~


# In[16]:


dfm = dfb.merge(df, how='left', on='datetime')


# In[17]:


dfm.describe()


# In[23]:


dfm.head(62)


# In[ ]:


dfm = dfm[columnslist]


# In[ ]:


# dfm = dfm[dfm['sp_percent_change_daily'] != 0]


# In[ ]:


dfm = dfm.interpolate(method ='linear')


# In[ ]:


# dfm_csv_fn = "./data/masterdf_hourly.csv"
# dfm.to_csv(dfm_csv_fn, index=False)


# In[25]:


dfm = pd.read_csv('/home/ubuntu/notebooks/deeplearning/project/data/masterdf_hourly_final_reduced.csv')


# In[26]:


columnslist = list(dfm.columns)
columnslist.remove('net-equity-expansion-A')
columnslist.remove('stablecoin-ratio-usd')
columnslist.remove('s&p 20day price moving average')
columnslist.remove('s&p volume 20day moving average')
columnslist.remove('gold price 20day moving average')
columnslist.remove('miner-outflow-mean')
columnslist.remove('stablecoin-supply-ratio')
columnslist.remove('addresses-count-total')
columnslist.remove('miner-inflow-mean')
columnslist.remove('miner-trans-inflow')
columnslist.remove('exchange-trans-count-reserve-inflow')
columnslist.remove('miner-trans-outflow')
columnslist.remove('stock-to-flow-reversion')
columnslist.remove('otc-volume')
columnslist.remove('miner-revenue')
columnslist.remove('coin-20day-std')
columnslist.remove('puell')
columnslist.remove('taker-buy-sell-stats-A')
columnslist.remove('blockreward-usd')
columnslist.remove('high')
columnslist.remove('low')
columnslist.remove('vwap')
columnslist.remove('mvrv')
columnslist.remove('miner-reserve-usd (onchain-balance)')
columnslist.remove('market-capitalization-usd')
columnslist.remove('funding-rates-A')
columnslist.remove('open-interest')
columnslist.remove('fee-trans')
columnslist.remove('fees')
columnslist.remove('otc-high')
columnslist.remove('bitcoin-price-ma20')


# In[27]:


dff = dfm[columnslist]


# In[28]:


dff.head()


# In[30]:


dff.describe()


# In[31]:


dff_csv_fn = "./data/masterdf_hourly_final_reduced_actual.csv"
dff.to_csv(dff_csv_fn, index=False)


# In[32]:


dfft = pd.read_csv('/home/ubuntu/notebooks/deeplearning/project/data/masterdf_hourly_final_reduced_actual.csv')


# In[33]:


dfft.describe()


# In[ ]:


# ~~~~ perform granger causality analysis ~~~~


# In[ ]:


columnslist = list(dfm.columns)
columnslist.remove('datetime')
features = dfm[columnslist]
# features = features[0:720]

transform_data = features.diff().dropna()
transform_data.head()


# In[ ]:


features.describe()


# In[ ]:


def augmented_dickey_fuller_statistics(i, time_series):
    result = adfuller(time_series.values)
    print('~~~~' + str(columnslist[i]) + '~~~~')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    print('')


# In[ ]:


for i in range(len(columnslist)):
    augmented_dickey_fuller_statistics(i, transform_data.iloc[:,i])


# In[ ]:


# CONCLUSION: all variables are stationary


# In[ ]:


maxlag= 24
test = 'ssr-chi2test'

def grangers_causality_matrix(X_train, variables, test = 'ssr_chi2test', verbose=True):
    
    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    nph = []
    
    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(X_train[[r,c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose and r == 'close': print(f'Y = {r}, X = {c}, P Values = {p_values}')
            if verbose and r == 'close': nph.append(p_values)
            min_p_value = np.min(p_values)
            dataset.loc[r,c] = min_p_value
    arr = np.array(nph)
    return dataset, nph


# In[ ]:


results, nph = grangers_causality_matrix(features, variables = features.columns)


# In[ ]:


results.describe()


# In[ ]:


gr_csv_fn = "grangercausality_close_hourly.csv"
results.to_csv(gr_csv_fn, index=False)


# In[ ]:


grpc_csv_fn = "closegranger_hourly.csv"
np.savetxt(grpc_csv_fn, nph, delimiter=",")


# In[ ]:


# ~~~~ plot datasets ~~~~


# In[ ]:


# dfs = pd.DataFrame(columns = columnslist, data = features)
# dfs.reset_index()


# In[ ]:


# columnslistplot = list(dfs.columns)


# In[ ]:


i = 6
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 6
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 7
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 8
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 9
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 10
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 11
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 12
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 13
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 14
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 15
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 16
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 17
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 18
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 19
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 20
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 21
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 22
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 23
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 24
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 25
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 26
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 27
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 28
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 29
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 30
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 31
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 32
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 33
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 34
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 35
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 36
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 37
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 38
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 39
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 40
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 41
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 42
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 43
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 44
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 45
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 46
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 47
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 48
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 49
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 50
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 51
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 52
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 53
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 54
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 55
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 56
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 57
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 58
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 59
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 60
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 61
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 62
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 63
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 64
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 65
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 66
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 67
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 68
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 69
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 70
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 71
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 72
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 73
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 74
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:


i = 75
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')


# In[ ]:




