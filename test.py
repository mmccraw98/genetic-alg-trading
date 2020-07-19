import numpy as np
import pandas as pd
from tree import tree, signal_function, node
import pickle
import matplotlib.pyplot as plt

root = 'C:\\Users\\PC\\PycharmProjects\\Finance\\venv\\generations\\'

mroi = []
mprod = []

broi = []
bprod = []

for gen in ['gen16', 'gen17', 'gen18', 'gen20', 'gen21', 'gen22', 'gen23', 'gen24', 'gen25']:
    gen_root = root+gen+'\\'

    with open(gen_root+'Generation Data\\Statistics\\gen26.stat', 'rb') as f:
        model = pickle.load(f)['best_model']

    n_random_stocks = ['AAPL',
              'MSFT',
              'AMZN',
              'GOOG',
              'FB',
              'BRKS',
              'BRKR',
              'BRKL',
              'V',
              'JNJ',
              'WMT',
              'MA',
              'PG',
              'UNH',
              'JPM',
              'HD',
              'INTC',
              'NVDA',
              'VZ',
              'TSLA',
              'T',
              'ADBE',
              'NFLX',
              'PYPL',
              'DIS',
              'BAC',
              'MRK',
              'KO',
              'CSCO',
              'PFE',
              'XOM',
              'PEP',
              'CMCSA',
              'ABBV',
              'CRM',
              'ORCL',
              'CVX',
              'ABT',
              'LLY',
              'NKE',
              'AMGN',
              'TMO',
              'ACN',
              'MCD',
              'COST',
              'BMY',
              'DHR',
              'AVGO',
              'MDT',
              'NEE',
              'AMT',
              'LIN']
    path = 'C:\\Users\\PC\\PycharmProjects\\Finance\\venv\\Data\\fin_base\\'

    # biggest ~50 stocks
    dfs = [pd.read_csv(path+stock+'.csv')[:365] for stock in n_random_stocks]

    # individual company
    #dfs = [pd.read_csv(path+'BMRA.csv')[:365]]

    # training basis
    #with open(gen_root+'Training Basis\\training_stocks.unvr', 'rb') as f:
    #    dfs = [pd.read_csv(path+stock+'.csv') for stock in pickle.load(f)]


    rois = []
    d_rois = []

    for df in dfs:
        model.predict(df)
        roi = 1
        owned = False
        for i, [buy, price] in enumerate(zip(model.prediction, df.Close)):
            if buy and not owned:
                owned = True
                buy_price = price
            elif not buy and owned:
                owned = False
                roi *= 1 + (price - buy_price) / buy_price
            elif i == df.shape[0] - 1 and owned:
                owned = False
                roi *= 1 + (price - buy_price) / buy_price
        if roi != 0:
            rois.append(1 - roi)
            d_rois.append((df.Close.values[-1] - df.Close.values[0]) / df.Close.values[0])
        #print(1 - roi, (df.Close.values[-1] - df.Close.values[0]) / df.Close.values[0])

    #print(20*'-')
    mroi.append(np.mean(rois))
    mprod.append(np.prod(rois))
    broi.append(np.mean(d_rois))
    bprod.append(np.prod(d_rois))
    print(np.mean(rois), np.mean(d_rois))
    print(np.prod(rois), np.prod(d_rois))
    print(np.sum(rois), np.sum(d_rois))

plt.plot(mroi, label='Model ROI')
#plt.plot(broi, label='Base ROI')
plt.legend()
plt.show()

plt.plot(mprod, label='Model Prod.')
#plt.plot(bprod, label='Base Prod.')
plt.legend()
plt.show()
