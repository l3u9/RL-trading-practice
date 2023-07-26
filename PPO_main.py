import pandas as pd
from PPO_Environment import *
from model import *
from utils import *

csvs = get_csv_path()

df = pd.DataFrame()
for csv in csvs:
    tmp = pd.read_csv(csv)
    df = pd.concat([df, tmp], axis=0)

df = df.sort_values('open_time')

cols = (df.columns[6:])


df = df.drop(cols, axis=1)
df = df.reset_index()


env = CustomEnv(df,lookback_window_size=1000)

train_agent(env, visualize=False, train_episodes=500, training_batch_size=1000)

# Random_games(env, False)