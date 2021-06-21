from find_sell import *
from find_buy import *

def buy_stock(price):
  return price*0.998

def sell_stock(price):
  return price*0.998

def get_next_buy_price(data, i):
  df = data[i:]
  price = df[df['action'] == 'buy'].head(1)['price'].values[0]
  index = df[df['action'] == 'buy'].head(1).index.values[0]
  date = df[df['action'] == 'buy'].head(1)['date'].values[0]
  return price, date, index

def get_next_sell_price(data, i):
  df = data[i:]
  price = df[df['action'] == 'sell'].head(1)['price'].values[0]
  index = df[df['action'] == 'sell'].head(1).index.values[0]
  date = df[df['action'] == 'sell'].head(1)['date'].values[0]
  return price, date, index

def get_next_buy_point(sell_point, check_low_list):
    i = 0
    j = len(check_low_list) - 1
    res = j
    if check_low_list[j] <= sell_point:
        return 0
    while i <= j:
        mid = (i + j) // 2
        if sell_point < check_low_list[mid]:
            res = mid
            j = mid - 1
        else:
            i = mid + 1

    return check_low_list[res]

def get_transaction_df(data, check_high_list, check_low_list):
    high = []
    for item in check_high_list:
        price = data.loc[item, ['open_price']].values[0]
        high.append((item, 'sell', price))

    low = []
    for item in check_low_list:
        price = data.loc[item, ['open_price']].values[0]
        low.append((item, 'buy', price))

    all = high + low

    df_transaction = pd.DataFrame(all, columns=['date', 'action', 'price'])
    df_transaction['date'] = pd.to_datetime(df_transaction['date'])
    df_transaction = df_transaction.sort_values('date').reset_index()[['date', 'action', 'price']]

    return df_transaction

def trans_agent(df_transaction, money=1000, print_=False):
    index = 0
    while index < len(df_transaction) - 1:

        buy_price, buy_date, index = get_next_buy_price(df_transaction, index)
        stock_num = buy_stock(money) / buy_price

        if print_:
            print('--- BUY: {0:10}        DATE: {1:10}'.format(stock_num, buy_date))

        index += 1
        sell_price, sell_date, index = get_next_sell_price(df_transaction, index)
        money = sell_stock(stock_num * sell_price)

        if print_:
            print('++ SELL: {0:10}        DATE: {1:10}'.format(money, sell_date))
            print()

    if print_:
        print("==================================================================================")
        print("==================================================================================")
        print("The money you earned is :", money - 1000)

    return money - 1000

def get_transaction_list(check_low_list, df, rebound_rate, deadline_rate, hold_time_limit, dr):
    buy_point_list, sell_point_list = [], []
    buy_point = check_buy_point(df, check_low_list[0], check_low_list, dr)

    while buy_point != 0:
        sell_point = find_sell_point(buy_point, df, rebound_rate, deadline_rate, hold_time_limit)
        if sell_point is not None:
            buy_point_list.append(buy_point)
            sell_point_list.append(sell_point)
        else:
            break
        next_point = get_next_buy_point(sell_point, check_low_list)
        buy_point = check_buy_point(df, next_point, check_low_list, dr)

    return sell_point_list, buy_point_list


