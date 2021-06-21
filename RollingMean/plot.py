import time
import datetime
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def plot_data(data, highlight=False, checklist=None):

    with plt.style.context('seaborn-dark'):
        plt.figure(figsize=(8, 4))
        plt.plot(data.close_price.values, label='open_price', linewidth = 1, alpha=0.4, color='#FFA500')
        plt.plot(data.open_price.values, label='close_price', linewidth = 1, alpha=0.7, color='#2c7fb8')

        if highlight is True:
          base = time.mktime(datetime.datetime.strptime(data.iloc[[0]].index.values[0],'%Y-%m-%d %H:%M:%S').timetuple())
          x_list = []
          for item in checklist:
            ts = time.mktime(datetime.datetime.strptime(item,'%Y-%m-%d %H:%M:%S').timetuple())
            idx = (ts - base) / 60
            x_list.append(int(idx))
          for item in x_list:
            plt.axvline(item, linewidth = 1.3, alpha=0.8, color='#d8b365')

        plt.title('Stock Price')
        plt.xlabel('minutes')
        plt.ylabel('price')
        plt.legend(loc='best')
        plt.show()

def get_sliced_df(data, s, e, df_transaction):

  d_test = data[['open_price', 'close_price']]
  d_test = d_test.reset_index()

  start_date = str(df_transaction.loc[s]['date'])
  end_date = str(df_transaction.loc[e]['date'])

  start_index = d_test[d_test['date'] == start_date].index.values[0] - 10
  end_index = d_test[d_test['date'] == end_date].index.values[0] + 10

  df_slice = d_test[start_index:end_index]
  df_slice = df_slice.set_index('date')

  highlight_points = [start_date, end_date]
  return df_slice, highlight_points

def plot_sliced_df(data, df_trans, N):
    for i in range(N):
        print(
            "======================================================================================================================================================================================================")
        print(
            "======================================================================================================================================================================================================")
        print(
            "======================================================================================================================================================================================================")
        print("The No.", (i + 1), "'th transaction:")
        df_temp, hl_temp = get_sliced_df(data, i * 2, i * 2 + 1, df_trans)
        plot_data(df_temp, highlight=True, checklist=hl_temp)