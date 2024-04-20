from finta import TA
import pandas as pd
from IPython.display import display
from random import uniform, randint
import matplotlib.pyplot as plt
from queue import Queue
from time import mktime
from datetime import datetime
import bs4 as bs
import pickle
import requests
import json
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import time
import os
from indicators import *


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 8)

# later we need to download CSV files of cryptos (Historical Datas)
# this variable determine the name of subFolder that CSVs wiil be saved inside that.
sub_folder_CSVs_store_location = 'Yahoo_Finance_CSV'



# <---------------------- Start: The Core Program ----------------------->
# to see how we use this go to line: 374


# this checks top 500 crypto symbles based on Coin Market Cap Ranking
# and extract the names but 'stable coins' are ignored
# the results will be saved in a TXT file with name: "top_500_crypto_names.txt"
# but it should be smaller than 500 cause 'stable coins' were ignored.
# Once we used this function, it won't be needed soon. cause changes in crypto ranking don't happen regularly.

def save_top_500_crypto_names():

    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start': '1',
        'limit': '500',
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': 'fe8952c1-2b22-40f5-ad0d-17ace1dce876',
    }

    session = Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)

    crypto_list = []

    for i in range(500):
        if 'stablecoin' not in data['data'][i]['tags']:
            sym_name = data['data'][i]['symbol']
            crypto_list.append(sym_name)

    with open("top_500_crypto_names.txt", "w") as f:
        for name in crypto_list:
            f.write(name + '\n')

    return crypto_list




# with this function we just load the names of top 500 cryptos that we already gathered from Coin Market Cap.

def load_top_500_crypto_names():

    crypto_list = []

    with open("top_500_crypto_names.txt", "r") as file:
        for line in file:
            crypto_list.append(line[:-1])

    return crypto_list



# Application: Download and Save Historical Data (Date, Open, High, Low, Close, Adjusted_Close, Value) of cryptos from Yahoo Finance
#     tickers_list: list crypto names (data type: list of strings)
#     p1: starting date (data type: datetime)
#     p2: ending date (data type: datetime)
#     interval: as default we use '1d' which means in CSV file each row will be a one day candlestick.

def get_yahoo_datasets(tickers_list, p1=datetime(2000, 1, 1), p2=datetime.now(), interval='1d'):
    # if your dataset is outdated or you don't have any dataset, you can use this to download the csv file
    p1s = int(mktime(p1.timetuple()))
    p2s = int(mktime(p2.timetuple()))

    counter = 0
    counter_max = len(tickers_list)

    for ticker in tickers_list:

        query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}-USD?period1={p1s}&period2={p2s}&interval={interval}&events=history&includeAdjustedClose=true'
        df = pd.read_csv(query_string, index_col='Date')
        df.to_csv(f'./{sub_folder_CSVs_store_location}/{ticker}.csv')

        # req = requests.get(query_string)
        # url_content = req.content

        # with open(f'./Yahoo_Finance_CSV/{ticker}.csv', 'wb') as csv_file:
        #     csv_file.write(url_content)

        counter += 1
        if counter % 10 == 0:
            print(
                f'{int(counter/counter_max*100)}% Completed! ({counter}/{counter_max})')

        # Yahoo will ban if we frequently send requests
        # time.sleep(5)
    print(f'100% Completed! ({counter}/{counter_max})')



# Application: load one crypto historical data and return pandas data frame (DF)
#    dataset_name: name of crypto. like 'BTC, 'ETH',... (data type: string)
#    start: starting date (data type: string) - like '2008-04-25'
#    end: ending date (data type: string)

def load_dataset(dataset_name, start='2000-01-01', end=datetime.now().strftime('%Y-%m-%d')):
    # this will load the csv file of the asset you mentioned and return the pandas DF of it in given time frame you gave.
    # start and end should string. dataset_name should be the name only without '-USD'. just like 'BTC'.

    path = f'./{sub_folder_CSVs_store_location}/{dataset_name}.csv'
    df = pd.read_csv(path, index_col='Date')

    return df.loc[start:end]




# This class is used to keep track of all buy and sell trades that we had.
# it has a dictionary containing important informations like: 'buy price', 'buy date', 'sell_price', and so on.
# This will be used inside Algorithm Class.
# This dictionary at the end will be feeded to pandas read_csv function which returns a dataframe. it is called TH that stands for trading history.
class TradingHistory:

    def __init__(self):
        self.dict = {'buy_date': [], 'sell_date': [], 'buy_price': [
        ], 'sell_price': [], 'change': [], 'e_profit': [], 'DH': []}

    def add_buy_info(self, buy_price, buy_date):

        self.dict['buy_date'].append(buy_date)
        self.dict['buy_price'].append(format(buy_price, ".3f"))

    def add_sell_info(self, sell_price, sell_date, days_hold, profit, e_profit):

        self.dict['sell_date'].append(sell_date)
        self.dict['sell_price'].append(format(sell_price, ".3f"))
        self.dict['change'].append(float(format((profit-1)*100, ".1f")))
        self.dict['e_profit'].append(float(format(e_profit, ".2f")))
        self.dict['DH'].append(days_hold)



# In this calss, following 3 methods are defined empty and we should write the logic or train a machine learning model to decide when to buy and when to sell:
#    check_buy_condition: this should return a TRUE or FALSE. when it returns TRUE, the algorithm will buy at that moment.
#    check_sell_condition: same as previous descriptions.
#    evaluation: it is used for evaluate anything. for example we have specific strategy; we run the algorithm and then we want to find out how many of our trades had no profit.
#
# initialization: this method is used to load a new crypto symble into algorithm
#    ticker: name of one ticker like: 'BTC'
#    show: if it's 1, when algorithm running progress is done, then it will show some results like: number of trades, number of profitable trades, effective profit that we had from all trades, and so on. 
#    saveHistory: if it's 1, the trade history will be saved using TradingHistory Class we wrote above, Otherwise we will not have any trade history and just we know the trade effective profit.
#    start: start date (data type: string)
#    end: end date (data type: string)
#
# add_indicator: adding any indicator
#    type: name of indicator. like: 'ma', 'rsi', 'macd', and...
#    val_1, val_2, val_3: these inputs are used for setting up the indicators. for example a MA indicator need one value to determine specific number of days.
#
# run: run the algorithm.
#    start_day: default value is 30. it is used to determine the start day of algorithm. imagine Moving Average (MA) indicator. for example if use a 30 days MA. then we don't have
#               the values of MA until day 30. so algorithm cannot run until day 30. in this case if we set start day to 20, then we will get error.

class Algorithm:
    def __init__(self):
        # DF informations
        self.N = 0
        self.ticker = None
        self.close_prices = []
        self.open_prices = []
        self.low_prices = []
        self.high_prices = []
        self.date = []
        self.ohlc = None

        # Algorithm Variables
        self.TH = None
        self.df_trade_history = None
        self.e_profit = 0
        self.change = 0
        self.trade_fee = 0.0012
        self.latest_buy_price = 0
        self.money_free = 1
        self.days_hold = 0
        self.eval_val = 0

        # indicators
        # if we can have from an indicators multiple, then it should be a list otherwise a NonoType
        self.ma = []
        self.rsi = None
        self.macd = None
        self.signal = None

    def initialization(self, ticker, show=0, saveHistory=1, start='2000-01-01', end=datetime.now().strftime('%Y-%m-%d')):
        self.show = show
        self.saveHistory = saveHistory
        if saveHistory:
            self.TH = TradingHistory()
        else:
            self.TH = None
        self.df_trade_history = None

        self.ticker = ticker
        df = load_dataset(ticker, start, end)

        self.ohlc = df.loc[:, 'Open':'Close']
        self.ohlc.rename(columns={'Open': 'open', 'High': 'high',
                                  'Low': 'low', 'Close': 'close'}, inplace=True)

        self.N = int(df.shape[0])
        self.close_prices = list(df['Close'])
        self.open_prices = list(df['Open'])
        self.low_prices = list(df['Low'])
        self.high_prices = list(df['High'])
        self.date_arr = list(df.index)

    def __buy__(self, day):
        self.money_free = 0
        buy_price = self.open_prices[day]
        self.latest_buy_price = buy_price
        self.e_profit *= (1 - self.trade_fee)
        self.latest_buy_date = self.date_arr[day]

        # adding trade history
        if self.saveHistory:
            self.TH.add_buy_info(buy_price=buy_price,
                                 buy_date=self.latest_buy_date)

    def __sell__(self, price, date):

        self.money_free = 1

        self.change = (price / self.latest_buy_price) * \
            (1 - self.trade_fee)

        self.e_profit *= self.change

        # adding trade history
        if self.saveHistory:
            self.TH.add_sell_info(
                sell_price=price,
                sell_date=date,
                days_hold=self.days_hold,
                profit=self.change,
                e_profit=self.e_profit
            )

        # self.__check_evaluation_metric__()
        self.days_hold = 0

    def __clear_indicators__(self):
        # clear the indicators
        self.ma = []
        self.macd = None
        self.signal = None
        self.rsi = None

    def add_indicator(self, type, val_1=1, val_2=1, val_3=1):

        if type == 'sma':
            self.ma.append(
                MovingAverage(
                    N=val_1,
                    close_prices_arr=self.close_prices)
            )

        elif type == 'macd':
            self.macd, self.signal = MACD(close_prices_arr=self.close_prices)

        elif type == 'rsi':
            self.rsi = TA.RSI(self.ohlc)

    def check_buy_condition(self, day):
        pass

    def check_sell_condition(self, day):
        pass

    def evaluation(self):
        pass

    def __show_results__(self):
        print(f'\n<------------- {self.ticker} ------------->')
        print(f'--> Effective Profit: {format(self.e_profit, ".3f")}')
        if self.saveHistory:
            num_all_trades = self.df_trade_history.shape[0]
            print('# Trades:', num_all_trades)
            num_no_profitable_trades = self.df_trade_history[
                self.df_trade_history['change'] < 0].shape[0]
            print(f'# Trades with no profit: {num_no_profitable_trades}')
            print(
                f'# Profitable Trades: {num_all_trades-num_no_profitable_trades}')
            print('\n< -- Trades with more than 30% profit  -- >')
            display(
                self.df_trade_history[self.df_trade_history['change'] > 30])

    def __check_evaluation_metric__(self):
        if evaluation:
            self.eval_val += 1

    def run(self, start_day=30):

        self.change = 0
        self.e_profit = 1
        self.latest_buy_price = 0
        self.money_free = 1
        self.days_hold = 0
        self.eval_val = 0

        for day in range(start_day, self.N):

            # Buy Section
            if self.money_free:
                if self.check_buy_condition(day):
                    self.__buy__(day)

            # Sell Section
            if not self.money_free:
                if day == self.N - 1:
                    self.__sell__(
                        price=self.close_prices[day],
                        date='~'
                    )

                elif self.days_hold > 0 and self.check_sell_condition(day):
                    self.__sell__(
                        price=self.open_prices[day],
                        date=self.date_arr[day]
                    )

            # Update Section
            if self.money_free == 0:
                self.days_hold += 1

        if self.saveHistory:
            self.df_trade_history = pd.DataFrame(self.TH.dict)

        if self.show:
            self.__show_results__()

        self.__clear_indicators__()



# <----------------------------- End: The Core Program ----------------------------->


# <---------------------- Start: The Following is a Use Case ----------------------->


# 1. first we get top 500 crypto names
#    this should get commented for future running of program
ticker_list = save_top_500_crypto_names()


# 2. loading the names
ticker_list = load_top_500_crypto_names()


# 3. get CSVs from Yahoo finance
#    this should get commented for future running of program
get_yahoo_datasets(tickers_list=ticker_list)


# 4. Defining buy condition
def check_buy_condition(self, day):
    cond_1 = self.ma[0][day-1] < self.high_prices[day-1]
    # cond_2 = self.signal[day-1] > 0
    cond_3 = self.rsi[day-1] < self.rsi[day-2]
    cond_4 = self.ma[1][day-1] < self.high_prices[day-1]
    cond_5 = self.macd[day-1] > self.macd[day-2]
    return cond_1 and cond_3 and cond_4


# 5. Defining sell condition
def check_sell_condition(self, day):
    # cond_0 = self.signal[day-1] < 0
    cond_1 = self.ma[0][day-1] > self.close_prices[day-1]
    # cond_2 = self.rsi[day-2] > self.rsi[day-1] and self.rsi[day-2] > 90
    cond_3 = self.macd[day-1] < 0
    return cond_1


# 5. Defining evaluation metric
def evaluation(self):
    return self.change < 0


# 6. first of all we correct the Algorithm Class
Algorithm.check_buy_condition = check_buy_condition
Algorithm.check_sell_condition = check_sell_condition
Algorithm.evaluation = evaluation


# 7. create a Algorithm object. We only need to create the object once
algo = Algorithm()


# 8. Iitialize the algorithm object. For each new crypto name, we should use this method again on this object (Reinitialization)
algo.initialization(
    ticker='FTM',
    show=1,
    saveHistory=1,
    start='2022-01-01',
)


# 9. Adding Indicators
algo.add_indicator(type='sma', val_1=20)
algo.add_indicator(type='sma', val_1=10)
algo.add_indicator(type='macd')
algo.add_indicator(type='rsi')


# 10. Finnaly we just run it
#     in initialization if we had set show=1 then should get some result after running completes.
algo.run()

# <------------------------- End: The Above was a Use Case --------------------------->
