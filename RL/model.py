import pandas as pd

from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG

from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env import DummyVecEnv

from env_valid import *
from env_trade import *
from env_train import *

import warnings
warnings.filterwarnings("ignore")

def train_A2C(env_train, model_name, timesteps=25000):
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(f"model/{model_name}")
    return model

def train_DDPG(env_train, model_name, timesteps=10000):
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    model.save(f"model/{model_name}")
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    model.learn(total_timesteps=timesteps)
    model.save(f"model/{model_name}")
    return model

def get_validation_sharpe(iteration):
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)

    if df_total_value['daily_return'].std() == 0:
      sharpe = 0
    else:
      sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe

def DRL_validation(model, test_data, test_env, test_obs):
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)
    return


def DRL_prediction(trade_data, model, name, last_state, iter_num, rebalance_window, initial=True):
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data)):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data) - 2):
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state