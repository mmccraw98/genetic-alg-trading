import pandas as pd
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
import errno
import time
from pandas_datareader import data
from yahoo_fin import stock_info
import datetime
from supporting_functions import update_stock_data

print('Working. . .')
update_stock_data(path='Data\\Stocks\\All\\')
print('Done!')
