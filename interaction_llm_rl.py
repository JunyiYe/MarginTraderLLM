import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../FinRL-Library")
import os
import argparse
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C
from finrl.plot import get_baseline, backtest_stats
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from env.marginEnv_v1 import MarginTradingEnv
from env.agent import DRLAgent
from finrl.main import check_and_make_directories
from finrl.config import (
    INDICATORS,
)
from utils import *


# set up

seed = 0
set_random_seed(seed)

parser = argparse.ArgumentParser(description='Interaction of LLM trend score with RL')
parser.add_argument('--data',type = str, default = 'firm_news')
parser.add_argument('--llm',type = str, default = 'gpt-4o-2024-05-13')
args = parser.parse_args()

TRAINED_MODEL_DIR = 'models'
RESULTS_DIR = 'results/test'

check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])


########################### Data processing
print("==============Process price data===========")
DOW_30_TICKER = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'GS',
       'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
       'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'VZ', 'WBA', 'WMT']

print(DOW_30_TICKER)
TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2018-12-31'
TEST_START_DATE = '2019-01-01'
TEST_END_DATE = '2020-04-30'
TRADE_START_DATE = '2020-05-01'
TRADE_END_DATE = '2024-04-01'


os.makedirs('./data', exist_ok=True)
if os.path.exists('./data/price_data.csv'):
    processed = pd.read_csv('./data/price_data.csv',index_col=0)
else:
    df = YahooDownloader(start_date = TRAIN_START_DATE,
                        end_date = TRADE_END_DATE,
                        ticker_list = DOW_30_TICKER).fetch_data() # 97013,8

    df.sort_values(['date','tic']).head()

    fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list = INDICATORS,
                        use_vix=True,
                        use_turbulence=True,
                        user_defined_feature = False)

    processed = fe.preprocess_data(df)
    processed = processed.copy()
    processed = processed.fillna(0)
    processed = processed.replace(np.inf,0)
    processed.to_csv('./data/price_data.csv')

train = data_split(processed, TRAIN_START_DATE,TRAIN_END_DATE)
test = data_split(processed, TEST_START_DATE,TEST_END_DATE)
trade = data_split(processed, TRADE_START_DATE,TRADE_END_DATE)


# read prediction from LLMs
ratio = pd.read_csv(f"data/output/{args.data}_{args.llm}_greedy.csv").Prediction

# majority voting for 5 runs
# ratio = ratio.loc[:,['Prediction1', 'Prediction2', 'Prediction3', 'Prediction4',
#                      'Prediction5']].mode(axis=1)
# if ratio.shape[1]==1:
#     ratio = ratio[0]
# elif ratio.shape[1]==2:
#     ratio = np.where((~pd.isna(ratio[1]))&(abs(ratio[1])<abs(ratio[0])),ratio[1],ratio[0])
# elif ratio.shape[1]>2:
#     print('too much mode')
#     exit()

########################### Build model
stock_dimension = len(train.tic.unique())
state_space = 2*3 + 2*stock_dimension + len(INDICATORS)*stock_dimension # cash, long, short, 30 close, 30 holding shares, 30 tech
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension


env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": 2*stock_dimension,
    "reward_scaling": 1e-4,
    'penalty_sharpe': 0.05
}



############################## Trade without LLM
trained_a2c = A2C.load(TRAINED_MODEL_DIR+'/agent_a2c.pth')

print("==============Trade without LLM interaction===========")
e_trade_gym = MarginTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
trade_account_value_a2c, trade_actions_a2c, trade_state_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c,
    environment = e_trade_gym)

trade_actions_a2c.to_csv(RESULTS_DIR+"/trade_actions.csv")
trade_state_a2c.to_csv(RESULTS_DIR+"/trade_state.csv")

perf_stats = pd.DataFrame(backtest_stats(account_value=trade_account_value_a2c))
perf_stats.columns = ['wo_llm']

trade_result_a2c = trade_account_value_a2c.set_index(trade_account_value_a2c.columns[0])
equity_ratio = pd.DataFrame(trade_state_a2c.long_equity / (trade_state_a2c.long_equity + trade_state_a2c.short_equity))
equity_ratio.columns = ['wo_llm']

############################## Trade with LLM

print("==============Trade with LLM interaction===========")
processed_date = list(processed.date.unique())
scale = 0.1 # 10% of changes
freq = 6 # 6 months

ratio_freq = int(freq/3)
ratio_list = list(ratio*scale)[::ratio_freq]

dates= pd.date_range(TRADE_START_DATE, TRADE_END_DATE , freq='1M')-pd.offsets.MonthBegin(1)
dates = [str(i.date()) for i in dates]
dates = dates[::freq] + [TRADE_END_DATE]

trade_account_value_a2c_llm, trade_actions_a2c_llm, trade_state_a2c_llm  = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

for i in range(1, len(dates)):
    if i != len(dates) - 1:
        data = data_split_include(processed, dates[i - 1], dates[i])
        if dates[i] not in processed_date:
            end_date = data.date.iloc[-1]
            end_date = processed_date[processed_date.index(end_date) + 1]
            data = data_split_include(processed, dates[i-1], end_date)
    else:
        data = data_split(processed, dates[i - 1], dates[i])

    if i==1:
        target = 0.5*(1+ratio_list[i])
        # target = 0.5+ratio_list[i]
        env_kwargs['long_short_ratio'] = target/(1-target)
        kwargs = env_kwargs
    else:
        kwargs = {
                "hmax": 100,
                "initial": False,
                "previous_state": state.tolist(),
                "initial_amount": state[2]+state[5],
                "num_stock_shares": num_stock_shares,
                "buy_cost_pct": buy_cost_list,
                "sell_cost_pct": sell_cost_list,
                "state_space": state_space,
                "stock_dim": stock_dimension,
                "tech_indicator_list": INDICATORS,
                "action_space": 2*stock_dimension,
                "reward_scaling": 1e-4,
                'penalty_sharpe': 0.05
                }

    e_trade_gym = MarginTradingEnv(df = data, turbulence_threshold = 70,risk_indicator_col='vix', **kwargs)
    account_log, action_log, state_log = DRLAgent.DRL_prediction(
        model=trained_a2c,
        environment = e_trade_gym)

    if i != len(dates)-1:
        state, target_ratio = update_ratio(i, state_log.iloc[-1], ratio_list[i], stock_dimension, buy_cost_list, sell_cost_list)

        state_log.iloc[-1] = state
        account_log = account_log.iloc[:-1]

    trade_account_value_a2c_llm = pd.concat([trade_account_value_a2c_llm, account_log])
    trade_actions_a2c_llm = pd.concat([trade_actions_a2c_llm, action_log])
    trade_state_a2c_llm = pd.concat([trade_state_a2c_llm, state_log])

perf_stats_llm = pd.DataFrame(backtest_stats(account_value=trade_account_value_a2c_llm))
perf_stats_llm.columns = ['llm']
perf_stats = pd.concat([perf_stats, perf_stats_llm], axis=1)


trade_actions_a2c_llm.to_csv(RESULTS_DIR + "/trade_actions_llm_" + str(freq) + ".csv")
trade_state_a2c_llm.to_csv(RESULTS_DIR + "/trade_state_llm_" + str(freq) + ".csv")
trade_account_value_a2c_llm.to_csv(RESULTS_DIR + "/trade_account_llm_" + str(freq) + ".csv")



print("==============Compare to DJIA===========")
dji_df = get_baseline(
        ticker="^DJI",
        start = TRADE_START_DATE,
        end = TRADE_END_DATE)


df_dji = dji_df[["date", "close"]]
fst_day = df_dji["close"][0]
dji = pd.merge(
    df_dji["date"],
    df_dji["close"].div(fst_day).mul(1000000),
    how="outer",
    left_index=True,
    right_index=True,
).set_index("date")


res = trade_account_value_a2c_llm.set_index(trade_account_value_a2c_llm.columns[0])

result = pd.DataFrame(
    {
        "wo_llm": trade_result_a2c["account_value"],
        "w_llm": res["account_value"],
        "dji": dji["close"]
    }
)

result.to_csv(RESULTS_DIR + "/trade_result.csv")
perf_stats_dji = pd.DataFrame(backtest_stats(dji_df,'close'))
perf_stats_dji.columns = ['dji']
perf_stats = pd.concat([perf_stats_dji, perf_stats], axis=1)
perf_stats.to_csv(RESULTS_DIR+"/perf_stats.csv")

read_path = RESULTS_DIR + "/trade_result.csv"
save_path = RESULTS_DIR + "/equity.pdf"
plot_equity(read_path, save_path)