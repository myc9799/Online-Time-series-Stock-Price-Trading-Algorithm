def find_sell_point(buy_point, df, rebound_rate, deadline_rate, hold_time_limit):

  check_df = df.loc[buy_point:]
  threshold = df.loc[buy_point]['open_price']
  flag = 0
  flag2 = 0

  for index, rows in check_df.iterrows():

    flag2 += 1
    if flag2 >= hold_time_limit:
      sell_point = index
      return sell_point

    elif rows['open_price'] > threshold * (rebound_rate + 1):
      flag = 0
      sell_point = index
      return sell_point

    elif rows['open_price'] < threshold * (1-deadline_rate):
      flag += 1
      if flag >= 1440:
        sell_point = index
        return sell_point

  sell_point = check_df.tail(1).index.values[0]
  return sell_point