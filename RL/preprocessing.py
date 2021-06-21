# from stockstats import StockDataFrame as Sdf
# import pandas as pd

# all = pd.read_csv("train_data/2018_with_blanks.csv")
# all.columns = ['date', 'open', 'close', 'high', 'low']
# stock = Sdf.retype(all.copy())
# all['macd'] = pd.DataFrame(stock['macd']).reset_index()['macd']
# all['rsi'] = pd.DataFrame(stock['rsi_30']).reset_index()['rsi_30']
# all['adx'] = pd.DataFrame(stock['dx_30']).reset_index()['dx_30']
# all['cci'] = pd.DataFrame(stock['cci_30']).reset_index()['cci_30']


# trading = all.iloc[216320:]
# train = all[:171957]
# valid = all[171957:216320]

# test = valid.copy()
# test = test.reset_index()
# test = test.drop(columns=['index'])
# pivot = test[['date', 'close']]

# count = 0
# turbulence_index = [0]

# for i in tqdm(range(len(pivot))):

#   current_price = pivot.loc[i].close
#   hist_price = pivot[0:i]
#   cov_temp = hist_price.cov()
#   current_temp = (current_price - np.mean(hist_price,axis=0))

#   if i > 2:
#     temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
#   else:
#     temp = 0

#   if temp > 0:
#     count += 1
#     if count > 2:
#       turbulence_temp = temp
#     else:
#       turbulence_temp = 0
#   else:
#     turbulence_temp = 0

#   turbulence_index.append(turbulence_temp)

# turbulence_index = np.array(turbulence_index)
# np.save('DATA/turbulence_index', turbulence_index)

# trading.to_csv('trade_half.csv', index=False)
# train.to_csv('train_half.csv', index=False)
# valid.to_csv('valid_half.csv', index=False)