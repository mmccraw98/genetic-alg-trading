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
import base64
from email.mime.text import MIMEText
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
from math import floor
#from googleapiclient.discovery import build

vader = SentimentIntensityAnalyzer()
# New words and values
new_words = {
    'crushes': 10,
    'beats': 5,
    'misses': -5,
    'trouble': -10,
    'falls': -100,
}
# Update the lexicon
vader.lexicon.update(new_words)

def save_obj_to_path(obj, path, name):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path + name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj_from_path(path, name):
    try:
        with open(path + name, 'rb') as f:
            return pickle.load(f)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def get_folders(path):
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]


def get_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def make_folders(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


def load_stock_data(path, name):
    '''
    Loads saved stock data into a dataframe, typically for investigative use, should not be used in important programs
    :param path: string, must  be denoted as path\\to\\main\\folder\\
    :param name: string, will be the name of the .csv  file loaded, must not have any file denotion in it
    :return: loads the stock data into a dataframe
    '''
    return pd.read_csv(path+name+'.csv', index_col=0)


def make_training_dir(path, dirname):
    root = path + '\\' + dirname + '\\'
    make_folders(root)
    #make_folders(root + 'Universe\\')
    make_folders(root + 'Generations\\')


def get_recent(path):
    gen_numbers = [int(file.split(sep='_')[-1]) for file in get_files(path=path)]
    return path + 'Gen_' + str(max(gen_numbers))


def get_time_report(stats_dict, params_dict):
    curr_gen = params_dict['current_gen']
    num_generation = params_dict['numgen']
    avggentime = np.mean(stats_dict['gen_run_time'])
    start_time = stats_dict['start_time']
    timeremaining = avggentime * (int(num_generation) - int(curr_gen))
    if timeremaining >= 60:  # minutes
        if timeremaining >= 60 * 60:  # hours
            if timeremaining >= 60 * 60 * 24:  # days
                timeremaining /= (60 * 60 * 24)
                timeremaining = '{:.2f} d'.format(timeremaining)
            else:
                timeremaining /= (60 * 60)
                timeremaining = '{:.2f} h'.format(timeremaining)
        else:
            timeremaining /= 60
            timeremaining = '{:.2f} m'.format(timeremaining)
    else:
        timeremaining = '{:.2f} s'.format(timeremaining)

    if avggentime >= 60:  # minutes
        if avggentime >= 60 * 60:  # hours
            if avggentime >= 60 * 60 * 24:  # days
                avggentime /= (60 * 60 * 24)
                avggentime = '{:.2f} d'.format(avggentime)
            else:
                avggentime /= (60 * 60)
                avggentime = '{:.2f} h'.format(avggentime)
        else:
            avggentime /= 60
            avggentime = '{:.2f} m'.format(avggentime)
    else:
        avggentime = '{:.2f} s'.format(avggentime)

    totaltime = time.time() - start_time
    if totaltime >= 60:  # minutes
        if totaltime >= 60 * 60:  # hours
            if totaltime >= 60 * 60 * 24:  # days
                totaltime /= (60 * 60 * 24)
                totaltime = '{:.2f} d'.format(totaltime)
            else:
                totaltime /= (60 * 60)
                totaltime = '{:.2f} h'.format(totaltime)
        else:
            totaltime /= 60
            totaltime = '{:.2f} m'.format(totaltime)
    else:
        totaltime = '{:.2f} s'.format(totaltime)

    return timeremaining, avggentime, totaltime


def get_stats_report(stats_dict):
    print('Total: [ Best Score {:.3f} | Avg. Best Score {:.3f} | Avg. Model Size {} ] Current Gen.: [ Best Score {:.3f} | Worst Score {:.3f} ]'.format(
        stats_dict['best_score_total'][-1], stats_dict['avg_best_score_total'][-1], stats_dict['avg_size_from_gen'][-1], stats_dict['best_score_from_gen'][-1], stats_dict['worst_score_from_gen'][-1]
    ))


def update_tickers(path, name):
    '''
    Gets a list of tickers from the yahoo finance database and saves the list to a .csv file under a given directory and name
    :param path: string, must  be denoted as path\\to\\main\\folder\\
    :param name: string, will be the name of the .csv  file saved, must not have any file denotion in it
    :return: saves a list of tickers to path\\name.csv
    '''
    sources, tickers = [stock_info.tickers_dow(), stock_info.tickers_nasdaq(), stock_info.tickers_other(), stock_info.tickers_sp500()], []
    for source in sources:
        for ticker in source:
            if ticker not in tickers:
                tickers.append(ticker)
    pd.DataFrame(tickers).to_csv(path+name+'.csv')


def fetch_tick_list(path, name):
    '''
    Fetches the saved ticker list as a list
    :param path: string, must  be denoted as path\\to\\main\\folder\\
    :param name: string, will be the name of the .csv  file saved, must not have any file denotion in it
    :return: list of tickers
    '''
    return pd.read_csv(path+name+'.csv').values[:,1]


def get_stock_data(path, tickers):
    '''
    Gets historic financial data of all the tickers in a given list and saves them as a series of .csv files
    :param path: string, must  be denoted as path\\to\\main\\folder\\
    :param tickers: list, of tickers that will have their data gathered by the program and will also act as the names of each .csv file
    :return: a set of .csv files under the names listed in the tickers list with data for each ticker in the list gathered by pdr
    '''
    for i, tick in enumerate(tickers):
        start_date = datetime.datetime.now() - datetime.timedelta(days=365*20+100)
        end_date = datetime.date.today()
        try:
            data.get_data_yahoo(tick, start=start_date, end=end_date).dropna().to_csv(path+tick+'.csv')
        except Exception as e:
            print(e)


def get_newsapi_vader(query, date_str, key, vader_instance=vader):
    '''
    Gets the news article titles from a certain date - up until now for a certain query and returns the average sentiment
    from a vader_instance as well as the number of news mentions
    :param query: str, typically a stock ticker, could be anything you want news articles about though
    :param date_str: str, string representing the date you want data to be collected from (typically current day)
    :param key: str, api access key
    :param vader_instance: nltk vader instance for sentiment analysis
    :return: float or NaN, representing sentiment values about the query for the given day
    '''
    # url for news api access
    url = 'https://newsapi.org/v2/everything?'
    # news api search parameters dict set to collect all the news possible
    parameters = {
        'q': query, # query phrase
        'pageSize': 100,  # maximum is 100
        'apiKey': key, # your own API key
        'language': 'en', # desired language
        'from': date_str # date from which data will be collected
    }
    # gets the json response
    response = requests.get(url, params = parameters).json()
    # for storage of sentiment values and article titles
    sentiment, collected_articles = [], []
    # loop through all articles
    for article in response['articles']:
        # get titles and dates of publications for each article found
        title = article['title']
        date = article['publishedAt'].split(sep='T')[0]
        # make sure the data is for the desired date and the title is not a duplicate
        if date == date_str and title not in collected_articles:
            # store the article title for future duplicate prevention
            collected_articles.append(title)
            # store the compound sentiment value from the vader instance
            sentiment.append(vader_instance.polarity_scores(title)['compound'])
    # if no articles were found, return Nan for sentiment and 0 for mentions
    if len(sentiment) == 0:
        return np.nan, 0
    # otherwiise return the average sentiment and the number of mentions
    else:
        return np.mean(sentiment), len(sentiment)


def update_stock_data(path):
    '''
    Updates all of the stock data for every file within a specified path - MUST BE RUN AFTER THE DAILY CLOSE OF THE MARKETS
    :param path: string, must  be denoted as path\\to\\main\\folder\\
    :return: a set of .csv files under the names listed in the tickers list with data for each ticker in the list gathered by pdr
    '''
    # loading news api keys
    with open('.\\keys\\news_api\\news_api_keys_list.pkl', 'rb') as f:
        keys = pickle.load(f)
    # defining the name of the vader news sentiment column and the news mentions column
    v_news_col, news_mentions_col = 'Vader News Sentiment', 'News API Mentions'
    # defining the number of news api calls we can make in a day
    api_call_daily_limit = 500
    # getting the stocks we want to update
    stocks = [f for f in listdir(path) if isfile(join(path, f))]
    # setting the start time for time estimation
    start_time = time.time()
    # getting the current date and converting to a string for use with news api data
    today = datetime.date.today().strftime('%Y-%m-%d')
    # looping over all the stocks we want to update
    for i, stock in enumerate(stocks):
        # accessing the old data and using the date column as the index column
        old_data = pd.read_csv(path+stock, index_col=0)
        # getting the last updated date and adding one so it can be used as a marker to get data
        last_updated = datetime.datetime.strptime(old_data.last_valid_index().split(sep=' ')[0],
                                                  '%Y-%m-%d') + datetime.timedelta(days=1)
        # selecting the correct key based off of news api calls
        key = keys[floor(i/api_call_daily_limit)]
        # if the last updated date is before today
        if last_updated < datetime.datetime.now() - datetime.timedelta(days=1):
            try:
                # get new data from yahoo finance using the last updated marker as most recent data day + 1
                new_data = data.get_data_yahoo(stock.split(sep='.csv')[0], start=last_updated, end=datetime.datetime.now())
                # getting the sentiment with vader sentiment analysis using the query as $STOCK
                sentiment, mentions = get_newsapi_vader(query='$' + stock.split(sep='.csv')[0], date_str=today, key=key)
                # if there is not a vader column in old_data
                if v_news_col not in old_data.keys():
                    # add a vader column to old data and fill with Nan
                    old_data[v_news_col] = np.nan
                # is there is not a news mentions column ion old data
                if news_mentions_col not in old_data.keys():
                    # add it and fill with Nan
                    old_data[news_mentions_col] = np.nan
                # add a vader column to new data and fill with Nan - necessary in case where new_data is longer than 1 row
                new_data[v_news_col] = np.nan
                # add a news mentions column to new_data and fill with Nan - necessary as explained above
                new_data[news_mentions_col] = np.nan
                # assign a value to vader column in new_data
                new_data.at[new_data.last_valid_index(), v_news_col] = sentiment
                # assign number of mentions to the new data
                new_data.at[new_data.last_valid_index(), news_mentions_col] = mentions
                # concatenate new_data and old_data correctly
                pd.concat([old_data, new_data], axis=0).to_csv(path+stock)
            # typical exception for a "dry link" where there is no more data
            except Exception as e:
                # pass the exception, there tends to be many of these dry links so it is better to not warn as it clogs the log
                pass  # print(e, 'No Data')
        # if the last updated date is not before today
        else:
            # pass the exception, would slow down progress warning users of something trivial like this
            pass  # print(stock, 'Already Up To Date')
        # time estimation logger
        print('Progress: {:.1f}% | Total Time: {:.1f}s'.format(100*i/len(stocks), time.time()-start_time), end='\r')


def create_message(sender, to, subject, message_text):
    """Create a message for an email.
      Args:
          sender: Email address of the sender.
          to: Email address of the receiver.
          subject: The subject of the email message.
          message_text: The text of the email message.
      Returns:
          An object containing a base64url encoded email object.
    """
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}


def send_message(service, user_id, message):
    """Send an email message.
      Args:
          service: Authorized Gmail API service instance.
          user_id: User's email address. The special value "me"
          can be used to indicate the authenticated user.
          message: Message to be sent.
      Returns:
          Sent Message.
    """
    message = (service.users().messages().send(userId=user_id, body=message)
               .execute())
    print('Message Id: %s' % message['id'])
    return message


def get_service(path):
    with open(rf'{path}', 'rb') as token:
        creds = pickle.load(token)
    service = build('gmail', 'v1', credentials=creds)
    return service
