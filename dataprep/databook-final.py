import pandas as pd
import datetime
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests

# ~~~~ load price 1-minute data and filter for hourly ~~~~
dfbh = pd.read_csv('/home/ubuntu/notebooks/deeplearning/project/data/btc-masterdata.csv')
val = dfbh['datetime'].loc[lambda x: x=='2019-04-01 00:00:00'].index.tolist()
dfb = dfbh[val[0]:]

dfb = dfb.reset_index()
dfb = dfb.drop(columns=['index'])
dfb['datetime'] = dfb['datetime'].astype('datetime64')
dfb = dfb[dfb.index % 60 == 0]
df = pd.read_csv('/home/ubuntu/notebooks/deeplearning/project/data/datafile2.csv')
datadf = pd.DataFrame({'year':df['year'],'month':df['month'],'day':df['day']})
df.insert(0, 'datetime', pd.to_datetime(datadf).dt.floor('D').dt.strftime('%Y-%m-%d %H:%M:%S'))
df = df.drop(columns=['year', 'month', 'day'])
df['datetime'] = df['datetime'].astype('datetime64')

# ~~~~ merge datasets (features and prices) ~~~~
dfm = dfb.merge(df, how='left', on='datetime')
dfm = dfm[columnslist]

# ~~~~ linear interpolation ~~~~
dfm = dfm.interpolate(method ='linear')

# ~~~~ write file ~~~~
# dfm_csv_fn = "./data/masterdf_hourly.csv"
# dfm.to_csv(dfm_csv_fn, index=False)

# ~~~~ test / read for features filtering ~~~~
dfm = pd.read_csv('/home/ubuntu/notebooks/deeplearning/project/data/masterdf_hourly_final_reduced.csv')

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

dff = dfm[columnslist]

# ~~~~ write file ~~~~
dff_csv_fn = "./data/masterdf_hourly_final_reduced_actual.csv"
dff.to_csv(dff_csv_fn, index=False)

# ~~~~ perform granger causality analysis ~~~~
columnslist = list(dfm.columns)
columnslist.remove('datetime')
features = dfm[columnslist]
# features = features[0:720]

# ~~~~ perform differencing ~~~~
transform_data = features.diff().dropna()
transform_data.head()

features.describe()

def augmented_dickey_fuller_statistics(i, time_series):
    result = adfuller(time_series.values)
    print('~~~~' + str(columnslist[i]) + '~~~~')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    print('')

for i in range(len(columnslist)):
    augmented_dickey_fuller_statistics(i, transform_data.iloc[:,i])

# CONCLUSION: all variables are stationary

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

results, nph = grangers_causality_matrix(features, variables = features.columns)

results.describe()

# ~~~~ write granger results ~~~~
gr_csv_fn = "grangercausality_close_hourly.csv"
results.to_csv(gr_csv_fn, index=False)

grpc_csv_fn = "closegranger_hourly.csv"
np.savetxt(grpc_csv_fn, nph, delimiter=",")

# ~~~~ plot datasets ~~~~

# dfs = pd.DataFrame(columns = columnslist, data = features)
# dfs.reset_index()

# columnslistplot = list(dfs.columns)

# change i-value
i = 6
ydata1 = dfs.iloc[:, 5]
ydata2 = dfs.iloc[:, i]
xdata = np.arange(len(dfs))

plt.title(columnslist[i])
plt.plot(xdata, ydata2, 'bo')
plt.plot(xdata, ydata1, 'go')
