import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import copy

def load_stock_data(path, name):
    '''
    Loads saved stock data into a dataframe, typically for investigative use, should not be used in important programs
    :param path: string, must  be denoted as path\\to\\main\\folder\\
    :param name: string, will be the name of the .csv  file loaded, must not have any file denotion in it
    :return: loads the stock data into a dataframe
    '''
    return pd.read_csv(path+name+'.csv', index_col=0)

# MAKE A MAJOR TREE BASED ALGORITHM
# THIS IS DIFFERENT THAN WHAT I AM DOING NOW
# ESSENTIALLY, WE WOULD NEED TO GROW AN ALGORITHM FOR SEVERAL SCENARIOS
# (VOLATILE-LOW CAP ALG, STABLE-HIGH CAP ALG, BEAR-TECH ALG, ETC...)
# THEN CAN STRUCTURE MAJOR TREE TO DETERMINE WHICH ALG TO USE

def SMA(series, periods):
    '''
    Simple Moving Average of a pd series
    Exponential Moving Average of a pd series
    Typically two or more moving averages of different periods are used in conjunction to detect changes in momentum
    I.E. if a short term moving average crosses over a long term moving average, the short term trend is increasing
    and the momentum is expected to increase, converse is true for a downwards cross
    Moving averages are calculated by taking the mean of the previous periods
    :param series: pandas series, note this has been generalized to be any series, but closing price is desired for this application
    :param periods: int, number of periods that the moving average will be calculated for: [1, length of series]
    :return: pandas series containing moving average of original series calculated over a number of periods, non-covered
    data will be returned as NaN
    '''
    return series.rolling(window=periods).mean()
def EMA(series, periods):
    '''
    Exponential Moving Average of a pd series
    Typically two or more moving averages of different periods are used in conjunction to detect changes in momentum
    I.E. if a short term moving average crosses over a long term moving average, the short term trend is increasing
    and the momentum is expected to increase, converse is true for a downwards cross
    Moving averages are calculated with an exponential weighted average of the previous periods so as to give
    exponentially less weight to older data and the most weight to the most recent data
    :param series: pandas series, note this has been generalized to be any series, but closing price is desired for this application
    :param periods: int, number of periods that the moving average will be calculated for: [1, length of series]
    :return: pandas series containing the exp. moving average of original series calculated over all periods
    '''
    return series.ewm(span=periods).mean()
def MACD(series, period_long, period_mid, period_signal):
    '''
    Moving Average Convergence Divergence of a pd series
    If the signal line crosses over the longer term line, upwards momentum is detected
    If the signal line crosses below the longer term line, downwards momentum is detected
    :param series:  pandas series, note this has been generalized to be any series, but closing price is desired for this application
    :param period_long: int, number of periods within the long term moving average
    :param period_mid: int, number of periods within the mid term moving average
    :param period_signal: int, number of periods within the signal line moving average
    :return: two pd series, typically mid term EMA - long term EMA and the EMA of that resulting signal
    Note: it is typical to have the structure listed above but, if designed algorithmically, it is not restricted
    for bivalued EMA to be of the form mid-long
    Investopedia recommends: Long:26, Medium: 12, Signal:9
    '''
    ema_bivalue = EMA(series=series, periods=period_mid) - EMA(series=series, periods=period_long)
    return ema_bivalue, EMA(series=ema_bivalue, periods=period_signal)
def BOLLINGER(series, periods, num_stds):
    '''
    Calculates the Bollinger bands for a pd series
    :param series: pandas series, note this has been generalized to be any series, but closing price is desired for this application
    :param periods: int, number of periods for the moving average calculation
    :param num_stds: int (or float), number of standard deviations above / below the standard deviation (width of bands)
    :return: three pd series, the simple moving average, the upper, and lower bollinger bands spaced by the standard deviations
    '''
    sma = SMA(series=series, periods=periods)
    std = num_stds * series.rolling(window=periods).std()
    return sma, sma + std, sma - std
def RSI(series, periods):
    '''
    Gives the Relative Strength Index of an asset
    RSI is a stochastic oscillator which can indicate whether an asset is overbought or oversold
    it will take values between 0 and 100.  Typically high values (>70%) are associated with overbought
    situations and lower values (<30%) are associated with oversold situations.  IMPORTANT TO NOTE THAT
    OVERBOUGHT / OVERSOLD THRESHOLDS TEND TO CHANGE AS THE TREND SHIFTS, I.E. IF THE PRICE HAS BEEN MOVING
    UP STEADILY, THE OVERBOUGHT THRESHOLD SHOULD BE ADJUSTED UPWARDS ACCORDINGLY
    :param series: pandas series, note this has been generalized to be any series, but closing price is desired for this application
    :param periods: int, number of periods for the moving average calculations
    :return: the relative strength index of the stock [0, 100]
    '''
    diffs = series.diff()
    ups, downs = diffs.copy(), diffs.copy()
    ups[ups < 0], downs[downs >= 0] = 0, 0
    sma_ups, sma_downs = SMA(series=ups, periods=periods), SMA(series=downs, periods=periods).abs()
    return 100 - 100 / (1 + sma_ups / sma_downs)
def DONCHIAN(series, periods):
    '''
    Computes the Donchian channels for a series
    The highest high and the lowest low within a defined period are plotted as well as their average or
    'mean reversion price'. The top channel shows the extent of the bullish action while the lower channel
    shows the extent of the bearish action and the middle channel shows the middle ground of the bear-bull
    action.
    :param series: pandas series, note this has been generalized to be any series, but closing price is desired for this application
    :param periods: int, number of periods over which the donchian channels will be computer
    :return: upper, middle, and lower donchian channels for a series over the defined number of periods
    '''
    upper_channel = series.rolling(window=periods).max()
    lower_channel = series.rolling(window=periods).min()
    middle_channel = (upper_channel + lower_channel) / 2
    return upper_channel, middle_channel, lower_channel
def ATR(series_high, series_low, series_close, periods):
    '''
    Returns the Average True Range signal of a series
    ATR is a volatility indicator, highly volatile stocks have a higher ATR than lower volatility stocks.
    There is no scale for ATR so it should be used subjectively and compared against it's historic value
    to make meaningful indications.
    ATR can be used to signal when to enter/exit a trade, for instance, the chandelier exit is a technique
    where some multiple of ATR is subtracted from the highest high since entereing the trade to serve as a
    trailing exit point
    :param series_high: pandas series, daily high values
    :param series_low: pandas series, daily low values
    :param series_close: pandas series, daily close values
    :param periods: int, number of periods over which ATR will be averaged to define the signal
    :return: pandas series containing the ATR of a given signal
    '''
    series_close = series_close.shift()
    true_range = pd.concat([series_high-series_low,
                            abs(series_high-series_close),
                            abs(series_low-series_close)], axis=1).max(axis=1)
    return SMA(series=true_range, periods=periods)
def KELTNER(series, series_high, series_low, series_close, periods, num):
    '''
    Gives the Keltner channels for a pandas series
    If the channels are angled up or down, the trend is seen as moving up or down.  The likelihood that this is correct
    is reinforced if the price is above or below the upper or lower channels while the channels are angled up or down.
    If the price continuously reaches one channel and suddenly reaches the opposite channel, that is an indication that
    the recent trend is over despite what the angle of the channels indicates.
    If the price is oscillating up and down, the bottom channel is seen as the support while the top channel is seen
    as the resistance level.
    :param series: pandas series, note this has been generalized to be any series, but closing price is desired for this application
    :param series_high: pandas series, daily high values
    :param series_low: pandas series, daily low values
    :param series_close: pandas series, daily close values
    :param periods: int, number of periods over which the Keltner Channels will be averaged to define the signal
    :param num: int (or float), defines the width of the channels in terms of some NUM multiple of the ATR
    :return: pandas series, containing the Keltner Channels of a signal
    '''
    atr = ATR(series_high=series_high, series_low=series_low, series_close=series_close, periods=periods)
    middle_channel = EMA(series=series, periods=periods)
    upper_channel = middle_channel + num * atr
    lower_channel = middle_channel - num * atr
    return upper_channel, middle_channel, lower_channel

class signal_function(object):
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.signal_type = self.params_dict['signal']
        self.periods = self.params_dict['periods']
        self.period_long = self.params_dict['period_long']
        self.period_mid = self.params_dict['period_mid']
        self.num = self.params_dict['num']

    def macd_tree(self):
        print('MACD NOT WORKING YET')
        series_close = self.df.Close
        ema_bivalue = EMA(series=series_close, periods=self.period_mid) - EMA(series=series_close,
                                                                                      periods=self.period_long)
        return ema_bivalue, EMA(series=ema_bivalue, periods=self.periods)

    def get_attributes(self):
        signals = ['SMA', 'EMA', 'BOLLINGER UPPER', 'BOLLINGER LOWER', 'RSI', 'DONCHIAN UPPER', 'DONCHIAN LOWER',
                   'DONCHIAN MIDDLE', 'ATR', 'KELTNER UPPER', 'KELTNER LOWER', 'CLOSE', 'HIGH', 'LOW', 'VOLUME']
        print(signals[self.signal_type], self.params_dict)

    def get_value(self, df):
        if self.signal_type == 0:
            return self.sma_tree(df=df)
        elif self.signal_type == 1:
            return self.ema_tree(df=df)
        elif self.signal_type == 2:
            return self.bollinger_upper_tree(df=df)
        elif self.signal_type == 3:
            return self.bollinger_lower_tree(df=df)
        elif self.signal_type == 4:
            return self.rsi_tree(df=df)
        elif self.signal_type == 5:
            return self.donchian_upper_tree(df=df)
        elif self.signal_type == 6:
            return self.donchian_lower_tree(df=df)
        elif self.signal_type == 7:
            return self.donchian_middle_tree(df=df)
        elif self.signal_type == 8:
            return self.atr_tree(df=df)
        elif self.signal_type == 9:
            return self.keltner_upper_tree(df=df)
        elif self.signal_type == 10:
            return self.keltner_lower_tree(df=df)
        elif self.signal_type == 11:
            return self.close_tree(df=df)
        elif self.signal_type == 12:
            return self.high_tree(df=df)
        elif self.signal_type == 13:
            return self.low_tree(df=df)
        elif self.signal_type == 14:
            return self.volume_tree(df=df)

    def sma_tree(self, df):
        series_close = df.Close
        return series_close.rolling(window=self.periods).mean()

    def ema_tree(self, df):
        series_close = df.Close
        return series_close.ewm(span=self.periods).mean()

    def bollinger_upper_tree(self, df):
        series_close = df.Close
        sma = SMA(series=series_close, periods=self.periods)
        std = self.num * series_close.rolling(window=self.periods).std()
        return (sma + std)

    def bollinger_lower_tree(self, df):
        series_close = df.Close
        sma = SMA(series=series_close, periods=self.periods)
        std = self.num * series_close.rolling(window=self.periods).std()
        return (sma - std)

    # compares to a percent [0, 100]
    def rsi_tree(self, df):
        series_close = df.Close
        diffs = series_close.diff()
        ups, downs = diffs.copy(), diffs.copy()
        ups[ups < 0], downs[downs >= 0] = 0, 0
        sma_ups, sma_downs = SMA(series=ups, periods=self.periods), SMA(series=downs, periods=self.periods).abs()
        return (100 - 100 / (1 + sma_ups / sma_downs))

    def donchian_upper_tree(self, df):
        series_close = df.Close
        upper_channel = series_close.rolling(window=self.periods).max()
        return upper_channel

    def donchian_lower_tree(self, df):
        series_close = df.Close
        lower_channel = series_close.rolling(window=self.periods).min()
        return lower_channel

    def donchian_middle_tree(self, df):
        series_close = df.Close
        upper_channel = series_close.rolling(window=self.periods).max()
        lower_channel = series_close.rolling(window=self.periods).min()
        middle_channel = (upper_channel + lower_channel) / 2
        return middle_channel

    def atr_tree(self, df):
        series_high = df.High
        series_low = df.Low
        series_close = df.Close.shift()
        true_range = pd.concat([series_high-series_low,
                                abs(series_high-series_close),
                                abs(series_low-series_close)], axis=1).max(axis=1)
        return SMA(series=true_range, periods=self.periods)

    def keltner_upper_tree(self, df):
        series_high = df.High
        series_low = df.Low
        series_close = df.Close
        atr = ATR(series_high=series_high, series_low=series_low, series_close=series_close, periods=self.periods)
        middle_channel = EMA(series=series_close, periods=self.periods)
        upper_channel = middle_channel + self.num * atr
        return upper_channel

    def keltner_lower_tree(self, df):
        series_high = df.High
        series_low = df.Low
        series_close = df.Close
        atr = ATR(series_high=series_high, series_low=series_low, series_close=series_close, periods=self.periods)
        middle_channel = EMA(series=series_close, periods=self.periods)
        lower_channel = middle_channel - self.num * atr
        return lower_channel

    def close_tree(self, df):
        return df.Close

    def high_tree(self, df):
        return df.High

    def low_tree(self, df):
        return df.Low

    def volume_tree(self, df):
        return df.Volume

class node(object):
    def __init__(self, indicator=None, threshold=None):
        self.indicator = indicator
        self.threshold = threshold
        self.true_connection = None
        self.false_connection = None
        self.index_true = None
        self.index_false = None
        self.base_connection = None
        self.base_connection_type = None
        self.label = None
        self.prediction = None
    def randomize(self):
        self.indicator = signal_function(params_dict={'signal': np.random.randint(0, 15),
                                                      'periods': np.random.randint(10, 200),
                                                      'period_mid': np.random.randint(15, 50),
                                                      'period_long': np.random.randint(50, 300),
                                                      'num': np.random.randint(1, 4)})
        self.threshold = signal_function(params_dict={'signal': np.random.randint(0, 15),
                                                      'periods': np.random.randint(10, 200),
                                                      'period_mid': np.random.randint(15, 50),
                                                      'period_long': np.random.randint(50, 300),
                                                      'num': np.random.randint(1, 4)})
        self.label = True#np.random.choice([True, True, True, False])
    def calculate_index(self, df):
        self.index_true = self.indicator.get_value(df=df) > self.threshold.get_value(df=df)
        self.index_false = -self.index_true
    def is_root(self):
        return self.base_connection is None
    def has_free_connection(self):
        return self.true_connection is None or self.false_connection is None
    def get_attributes(self):
        self.indicator.get_attributes()
        self.threshold.get_attributes()

class tree(object):
    def __init__(self, root=node(), rand_gen=False, size=None):
        self.root = root
        self.node_list = [self.root]
        self.free_node_list = [self.root]
        self.branches = None
        self.prediction = None
        if rand_gen and size is not None:
            self.generate_random(num_nodes=size)
    def get_size(self):
        return len(self.node_list)
    def randomize(self):
        for n in self.node_list:
            n.randomize()
    def add_to_node_list(self, new_node):
        #if new_node not in self.node_list:
        self.node_list.append(new_node)
    def update_free_node_list(self):
        missing_nodes = []
        for n in self.node_list:
            if n.true_connection not in self.node_list and n.true_connection is not None:
                missing_nodes.append(n.true_connection)
            if n.false_connection not in self.node_list and n.false_connection is not None:
                missing_nodes.append(n.false_connection)
        for n in missing_nodes:
            self.node_list.append(n)
        self.free_node_list = [n for n in self.node_list if n.has_free_connection()]
    def connect_node_to_true(self, base_node, connected_node):
        base_node.true_connection = connected_node
        connected_node.base_connection = base_node
        connected_node.base_connection_type = True
        self.add_to_node_list(new_node=connected_node)
        self.update_free_node_list()
    def connect_node_to_false(self, base_node, connected_node):
        base_node.false_connection = connected_node
        connected_node.base_connection = base_node
        connected_node.base_connection_type = False
        self.add_to_node_list(new_node=connected_node)
        self.update_free_node_list()
    def random_connect_free(self, base_node, connected_node):
        direction = np.random.choice([True, False])
        if direction: # try to connect to the true branch first
            if base_node.true_connection is None:
                self.connect_node_to_true(base_node=base_node, connected_node=connected_node)
            else:
                self.connect_node_to_false(base_node=base_node, connected_node=connected_node)
        else:
            if base_node.false_connection is None:
                self.connect_node_to_false(base_node=base_node, connected_node=connected_node)
            else:
                self.connect_node_to_true(base_node=base_node, connected_node=connected_node)
    def form_branches(self):
        self.branches = []
        for n in self.free_node_list:
            current_object = n
            sub_index_list = [[current_object, current_object.base_connection_type]]
            while not current_object.is_root():
                current_object = current_object.base_connection
                sub_index_list.append([current_object, current_object.base_connection_type])
            self.branches.append(sub_index_list)
    def predict(self, df):
        for n in self.node_list:
            n.calculate_index(df=df)
        final_products = []
        for branch in self.branches:
            running_product = branch[0][0].index_true
            leaf_label = branch[0][0].label
            for n, label in branch[1:]:
                if label:
                    running_product = running_product & n.index_true
                else:
                    running_product = running_product & n.index_false
            if leaf_label:
                final_products.append(running_product)
            else:
                final_products.append(-running_product)
        running_product = final_products[0]
        for final_prod in final_products[1:]:
            running_product = running_product | final_prod
        self.prediction = running_product
    def generate_random(self, num_nodes):
        for i in range(num_nodes - 1):
            self.random_connect_free(base_node=np.random.choice(self.free_node_list),
                                     connected_node=node())
            self.update_free_node_list()
        self.randomize()
        self.form_branches()
    def random_change_node(self, selected_node):
        selected_node.randomize()
    def slightly_alter_indicator_parameters(self, selected_node):
        dict = selected_node.indicator.params_dict
        rand = np.random.randint(-100, 100, size=(5,)) / 100
        params = (rand * np.array([20, 10, 50, 2, 2])).astype(int)
        res = dict['periods'] + params[0]
        if res >= 10 and res <= 200:
            dict['periods'] = res
        res = dict['period_mid'] + params[1]
        if res >= 15 and res <= 50:
            dict['period_mid'] = res
        res = dict['period_long'] + params[2]
        if res >= 50 and res <= 300:
            dict['period_long'] = res
        res = dict['num'] + params[3]
        if res >= 1 and res <= 4:
            dict['num'] = res
    def slightly_alter_threshold_parameters(self, selected_node):
        dict = selected_node.threshold.params_dict
        rand = np.random.randint(-100, 100, size=(5,)) / 100
        params = (rand * np.array([20, 10, 50, 2, 2])).astype(int)
        res = dict['periods'] + params[0]
        if res >= 10 and res <= 200:
            dict['periods'] = res
        res = dict['period_mid'] + params[1]
        if res >= 15 and res <= 50:
            dict['period_mid'] = res
        res = dict['period_long'] + params[2]
        if res >= 50 and res <= 300:
            dict['period_long'] = res
        res = dict['num'] + params[3]
        if res >= 1 and res <= 4:
            dict['num'] = res
    def remove_random_node(self, selected_node):
        removed = False
        if selected_node in self.node_list and len(self.node_list) > 3:
            self.node_list.remove(selected_node)
            removed = True
        if selected_node.true_connection is not None and len(self.node_list) > 3:
            self.remove_random_node(selected_node.true_connection)
        if selected_node.false_connection is not None and len(self.node_list) > 3:
            self.remove_random_node(selected_node.false_connection)
        if removed:
            del selected_node
    def random_prune_tree(self):
        selected_node = 0
        while selected_node == 0 and len(self.node_list) > 3:
            random_selection = np.random.choice(self.node_list)
            if random_selection != self.root:
                selected_node = random_selection
        self.remove_random_node(selected_node=selected_node)
        self.update_free_node_list()
        self.form_branches()
    def add_random_node(self):
        selected_node = np.random.choice(self.free_node_list)
        new_node = node()
        new_node.randomize()
        self.random_connect_free(base_node=selected_node,
                                 connected_node=new_node)
        self.update_free_node_list()
        self.form_branches()
    def mutate(self):
        selection = np.random.randint(0,100)
        if selection < 22:
            self.slightly_alter_indicator_parameters(np.random.choice(self.node_list))
        elif selection < 44:
            self.slightly_alter_threshold_parameters(np.random.choice(self.node_list))
        elif selection < 60:
            self.slightly_alter_indicator_parameters(np.random.choice(self.node_list))
            self.slightly_alter_threshold_parameters(np.random.choice(self.node_list))
        elif selection < 77:
            self.random_change_node(np.random.choice(self.node_list))
        elif selection >= 77:
            self.add_random_node()
        # THE ISSUE WITH THE CODE IS THE PRUNING FUNCTION, IT REALLY NEEDS WORK
        #elif selection >= 92 and len(self.node_list) > 3:
        #    self.random_prune_tree()
    def clear_prediction(self):
        self.prediction = None
    def save_as(self, path, file_name):
        with open(path+file_name+'.tr', 'wb') as f:
            pickle.dump(self, f)
    def basic_simulation(self, df, restr_range=[0, -1], clear_prediction=True, freq_weighted_scoring=False):
        self.predict(df=df[restr_range[0]: restr_range[1]])
        roi = 1
        risk = []
        trades = 0
        owned, start_idx = False, 0
        def freq_weighted_scoring_function(trade_freq):
            power_tower = (np.exp(-trade_freq**trade_freq**trade_freq**trade_freq**
                                  trade_freq**trade_freq**trade_freq**trade_freq**
                                  trade_freq**trade_freq) - 1/np.e) / 0.24429707873548812
            return power_tower
        for i, buy_signal in enumerate(self.prediction):
            if buy_signal and not owned:
                buy_price = df.Close[i]
                owned = True
                start_idx = i
                trades += 1
            if (not buy_signal and owned) or (i == self.prediction.size - 1 and owned):
                roi *= 1 + (df.Close[i] - buy_price) / buy_price
                risk.append(np.std(df.diff(periods=1).Close[start_idx: i].astype('float32')))
                owned = False
        if clear_prediction:
            self.clear_prediction()
        trades = trades / df.size
        if len(risk) == 0 or risk == 0:
            risk = 10000000
        if freq_weighted_scoring:
            if np.mean(risk) == 0:
                return 0
            else:
                #print(roi, np.mean(risk), freq_weighted_scoring_function(trades))
                return roi / np.mean(risk) * freq_weighted_scoring_function(trades)
        else:
            return roi, np.mean(risk), trades, np.prod(1+df.Close[restr_range[0]: restr_range[-1]].pct_change())
#@ TODO, UPGRADE THE SIMULATION TO BE FASTER AND OUTPUT A TUPLE OF MORE INFORMATIVE SCORES
