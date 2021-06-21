from trade import *
from find_buy import *
from plot import *
from heplers import *
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    return parser

WINDOW_SIZE = 1440

def test(WINDOW_SIZE=WINDOW_SIZE, drop_rate=0.91, rebound_rate=0.1, deadline_rate=0.1, hold_time_limit=10080, dr=0.1):

    data = pd.read_csv("test_data/2019_with_filling.csv")
    data = data.set_index('date').sort_index()
    data['open_price_rm'] = data['open_price'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

    date_low, day_low = detect_drop('open_price', data, drop_rate)
    sudden_drop_list = get_sudden_drop_list(date_low, day_low)
    check_low_list = get_low_check_list(data, sudden_drop_list)

    sell_point_list, buy_point_list = get_transaction_list(check_low_list, data, rebound_rate, deadline_rate, hold_time_limit, dr)
    df_transaction = get_transaction_df(data, sell_point_list, buy_point_list)
    revenue = trans_agent(df_transaction, print_=True)

    return revenue

def train(WINDOW_SIZE=WINDOW_SIZE, drop_rate=0.91, rebound_rate=0.1, deadline_rate=0.1, hold_time_limit=10080, dr=0.1):

    data = pd.read_csv("train_data/2018_with_filling.csv")
    data = data.set_index('date').sort_index()
    data['open_price_rm'] = data['open_price'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

    date_low, day_low = detect_drop('open_price', data, drop_rate)
    sudden_drop_list = get_sudden_drop_list(date_low, day_low)
    check_low_list = get_low_check_list(data, sudden_drop_list)

    sell_point_list, buy_point_list = get_transaction_list(check_low_list, data, rebound_rate, deadline_rate, hold_time_limit, dr)
    df_transaction = get_transaction_df(data, sell_point_list, buy_point_list)
    revenue = trans_agent(df_transaction, print_=True)

    return revenue

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()