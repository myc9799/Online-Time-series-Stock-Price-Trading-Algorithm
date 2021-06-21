import datetime
import itertools
import numpy as np
import pandas as pd

def detect_drop(str_, data, rate=0.9):

  price = str_
  price_rm = str_+'_rm'
  df = data[[price, price_rm]]
  df['drop'] = df.apply(lambda x: x[0] < x[1]*rate, axis=1)
  res = df[df['drop']==True].index.to_list()

  res_day = []
  for item in res:
    res_day.append(item[:10])
  res_day = list(set(res_day))
  res_day.sort()

  return res,res_day


# find the lowest point in a time period
def find_low(data, rate=0.98, lr=0.001):
    df = data[['open_price']]
    df['open_price_rm'] = data['open_price'].rolling(window=3, min_periods=1).mean()
    df['drop'] = df.apply(lambda x: x[0] < x[1] * rate, axis=1)
    res = df[df['drop'] == True].index.to_list()

    if len(res) == 0:
        res = find_low(data, rate + lr)
    return res

# find lower points and wait for further checking
def get_low_check_list(data, sudden_drop_list):
    check_list = []
    for i in range(len(sudden_drop_list)):
        df = data.loc[sudden_drop_list[i]]
        check_list.append(find_low(df))
    check_list = list(itertools.chain.from_iterable(check_list))

    return check_list

def get_sudden_drop_list(date, day):

  sudden_drop_list = []
  for days in day:
    col = []
    for item in date:
      if item[:10] == days:
        col.append(item)
    sudden_drop_list.append(col)

  return sudden_drop_list

def check_buy_point(data, p, cll, rate):
    if p == 0:
        return 0
    d = data.loc[get_time_period(p, 1440): p]
    d.index = pd.to_datetime(d.index)
    dd = d.resample('120T').mean()
    down_sampling_arr = dd['open_price'].to_list()
    flag = decrease_metric(down_sampling_arr, rate)

    if flag:
        idx = cll.index(p) + 1
        if idx < len(cll):
            p = cll[idx]
            res = check_buy_point(data, p, cll, rate)
        else:
            return p
    else:
        return p

    return res

def get_time_period(s, m):
    period = datetime.timedelta(minutes=m)
    temp = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    return (temp - period).strftime("%Y-%m-%d %H:%M:%S")

def decrease_metric(arr, rate):
    arr = np.array(arr)
    arr_diff = []
    for i in range(len(arr) - 1):
        arr_diff.append(arr[i + 1] - arr[0])

    arr_diff = arr_diff / arr[0]

    if True in (arr_diff < rate * -1.0):
        return True
    else:
        return False