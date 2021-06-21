from env_valid import *
from env_trade import *
from env_train import *
from model import *

import warnings
warnings.filterwarnings("ignore")

trade = pd.read_csv('data/trade_5.csv')
train = pd.read_csv('data/train_5.csv')
valid = pd.read_csv('data/valid_2.csv')

rebalance_window = 1440
validation_window = 1440
length = len(trade) + len(valid)

last_state_ensemble = []
ppo_sharpe_list = []
ddpg_sharpe_list = []
a2c_sharpe_list = []
model_use = []

for i in range(rebalance_window + validation_window, length, rebalance_window):

    # enviroment set up
    env_train = DummyVecEnv([lambda: StockEnvTrain(train)])
    env_val = DummyVecEnv([lambda: StockEnvValidation(df=valid, iteration=i)])
    obs_val = env_val.reset()

    # training and validating
    print("======A2C Training========")
    model_a2c = train_A2C(env_train, model_name="A2C_{}".format(i))
    DRL_validation(model=model_a2c, test_data=valid, test_env=env_val, test_obs=obs_val)
    sharpe_a2c = get_validation_sharpe(i)

    print("======PPO Training========")
    model_ppo = train_PPO(env_train, model_name="PPO_{}".format(i))
    DRL_validation(model=model_ppo, test_data=valid, test_env=env_val, test_obs=obs_val)
    sharpe_ppo = get_validation_sharpe(i)

    print("======DDPG Training========")
    model_ddpg = train_DDPG(env_train, model_name="DDPG_{}".format(i))
    DRL_validation(model=model_ddpg, test_data=valid, test_env=env_val, test_obs=obs_val)
    sharpe_ddpg = get_validation_sharpe(i)

    ppo_sharpe_list.append(sharpe_ppo)
    a2c_sharpe_list.append(sharpe_a2c)
    ddpg_sharpe_list.append(sharpe_ddpg)

    # model selecting
    if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
        model_ensemble = model_ppo
        model_use.append('PPO')
    elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
        model_ensemble = model_a2c
        model_use.append('A2C')
    else:
        model_ensemble = model_ddpg
        model_use.append('DDPG')

    # trading
    last_state_ensemble = DRL_prediction(trade_data=trade,
                                         model=model_ensemble,
                                         name="ensemble",
                                         last_state=last_state_ensemble,
                                         rebalance_window=rebalance_window,
                                         iter_num=i,
                                         initial=True)
