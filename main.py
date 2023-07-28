import pandas as pd
from Environment import *
from utils import *
from DDQN import *

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

# train_double_dqn(
#         env=env,
#         episodes=500,
#         batch_size=288,
#         gamma=0.99,
#         min_epsilon=0.01,
#         max_epsilon=1.0,
#         epsilon_decay=0.01,
#         learning_rate=1e-3,
#         target_update_freq=10,
#     )


# Initialize the agent
agent = DDQNAgent(env.state_size, 3)

# Set training parameters
episodes = 300
batch_size = 32

# Start training
train_ddqn(env, agent, episodes, batch_size)

# Close the environment when finished
env.close()