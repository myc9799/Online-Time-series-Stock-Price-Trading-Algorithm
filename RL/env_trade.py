import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000
STOCK_DIM = 1
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4
TURBULENCE = 10


class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, turbulence_threshold=TURBULENCE
                 , initial=True, previous_state=[], model_name='', iteration=''):

        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,))

        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     [self.data.close] + \
                     [0] * STOCK_DIM + \
                     [self.data.macd] + \
                     [self.data.rsi] + \
                     [self.data.cci] + \
                     [self.data.adx]

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        # self.reset()
        self._seed()
        self.model_name = model_name
        self.iteration = iteration

    def _sell_stock(self, index, action):

        if self.turbulence < self.turbulence_threshold:
            if self.state[index + STOCK_DIM + 1] > 0:
                # update balance
                self.state[0] += \
                    self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * \
                    (1 - TRANSACTION_FEE_PERCENT)

                self.state[index + STOCK_DIM + 1] -= min(abs(action), self.state[index + STOCK_DIM + 1])
                self.cost += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * \
                             TRANSACTION_FEE_PERCENT
                self.trades += 1
            else:
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions
            if self.state[index + STOCK_DIM + 1] > 0:
                # update balance
                # print(index)
                self.state[0] += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * \
                                 (1 - TRANSACTION_FEE_PERCENT)
                self.state[index + STOCK_DIM + 1] = 0
                self.cost += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * \
                             TRANSACTION_FEE_PERCENT
                self.trades += 1
            else:
                pass

    def _buy_stock(self, index, action):
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index + 1]

            # update balance
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * \
                             (1 + TRANSACTION_FEE_PERCENT)

            self.state[index + STOCK_DIM + 1] += min(available_amount, action)

            self.cost += self.state[index + 1] * min(available_amount, action) * \
                         TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_trade_{}_{}.png'.format(self.model_name, self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            print("previous_total_asset:{}".format(self.asset_memory[0]))

            print("end_total_asset:{}".format(end_total_asset))
            print("total_reward:{}".format(self.state[0] + sum(
                np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])) -
                                           self.asset_memory[0]))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)

            if df_total_value['daily_return'].std() == 0:
                sharpe = 0
            else:
                sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
                         df_total_value['daily_return'].std()
            print("Sharpe: ", sharpe)

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('results/account_rewards_trade_{}_{}.csv'.format(self.model_name, self.iteration))

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * HMAX_NORMALIZE
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)

            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                    self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :]

            self.turbulence = self.data.turbulence

            self.state = [self.state[0]] + \
                         [self.data.close] + \
                         self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)] + \
                         [self.data.macd] + \
                         [self.data.rsi] + \
                         [self.data.cci] + \
                         [self.data.adx]

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            self.asset_memory.append(end_total_asset)

            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                         [self.data.close] + \
                         [0] * STOCK_DIM + \
                         [self.data.macd] + \
                         [self.data.rsi] + \
                         [self.data.cci] + \
                         [self.data.adx]
        else:
            previous_total_asset = self.previous_state[0] + \
                                   sum(np.array(self.previous_state[1:(STOCK_DIM + 1)]) * np.array(
                                       self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            self.asset_memory = [previous_total_asset]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []

            self.state = [self.previous_state[0]] + \
                         [self.data.close] + \
                         self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)] + \
                         [self.data.macd] + \
                         [self.data.rsi] + \
                         [self.data.cci] + \
                         [self.data.adx]

        return self.state

    def render(self, mode='human', close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
