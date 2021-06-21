import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## read the csv file and rearrange the data
raw_data = pd.read_csv("2018.csv")
first = raw_data.columns.values
raw_data = raw_data.rename(columns={   '1514764800000':'date',
                            '736.77':'open_price',
                            '736.5':'close_price',
                            '736.82' :'highest_price',
                            '736.5.1':'lowest_price',
                            '45.36081317':'rate'})
first[4] = 736.51
first = pd.DataFrame(first).T
first = first.rename(columns={   0:'date',
                            1:'open_price',
                            2:'close_price',
                            3 :'highest_price',
                            4:'lowest_price',
                            5:'rate'})

data = first.append(raw_data)
data['date'] = pd.to_datetime(data['date'], unit='ms')
data = data.set_index('date')
data = data.drop(columns=['rate'])
data = data.astype(float)
data.to_csv('2018_with_blanks.csv')
def print_data_info(data):
    print('The info of the data is:')
    print(data.info())
    print('=========================================================================')
    print('=========================================================================')
    print('The statistics of the data is: \n', data.describe())

def check_missing_date(data):
    data_test = data.asfreq(freq='60S')
    missing_dates = data_test[data_test['open_price'].isnull()].index
    print("The missing dates are:\n", missing_dates)
    print("The number of missings is:",len(missing_dates.tolist()))
    return missing_dates

def plot_data(data):
    with plt.style.context('fivethirtyeight'):
        plt.figure(figsize=(15,7))
        plt.plot(data.highest_price.values, label='highest_price', linewidth=1, alpha=0.7)
        plt.plot(data.lowest_price.values, label='lowest_price', linewidth=1, alpha=0.7)
        plt.plot(data.open_price.values, label='open_price', linewidth = 1, alpha=0.7)
        plt.plot(data.close_price.values, label='close_price', linewidth = 1, alpha=0.7)
        plt.title('Stock Price')
        plt.xlabel('minutes')
        plt.ylabel('price')
        plt.legend(loc='best')
        plt.show()

def main():
    print('=========================================================================')
    print('=========================================================================')
    # missing_dates = check_missing_date(data)
    #
    # for item in missing_dates:
    #     data.loc[item] = np.nan
    #
    # data_no_missing = data.sort_index()
    # data_no_missing.fillna(axis=0, method='ffill', inplace=True)
    # data_no_missing= pd.read_csv('data_no_missing_2019.csv')
    print('=========================================================================')
    print('=========================================================================')
    # print_data_info(data_no_missing)
    print('=========================================================================')
    print('=========================================================================')
    # plot_data(data_no_missing)
    # data_no_missing.to_csv('data_no_missing_2019.csv')

main()