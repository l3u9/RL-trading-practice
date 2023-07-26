from datetime import datetime
import os
import numpy as np
from collections import deque

def train_ddqn(agent, env, episodes, timesteps, update_frequency):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(timesteps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

            if done:
                break

        if episode % update_frequency == 0:
            agent.update_target_net()

        print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")


def test_agent(env, visualize=True, test_episodes=10):
    env.load()
    average_net_worth = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            # env.render(visualize)
            action, prediction = env.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", episode, env.net_worth, env.episode_orders)
                break
            
    print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth/test_episodes))


    

def train_agent(env, visualize=False, train_episodes=50, training_batch_size=500):
    # env.create_writer()
    total_average = deque(maxlen=50)
    best_average = 0

    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            # env.render(visualize)
            # print(state.shape)
            action, prediction = env.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

        env.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)

        # env.writer.add_scalar('Data/average net_worth', average, episode)
        # env.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)

        print("net worth {} {:.2f} {:.2f} {}".format(episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                env.save()



def Random_games(env, visualize, train_episodes = 50, training_batch_size = 500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        while True:
            # env.render(visualize)
            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:",env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)


def Write_to_file(Date, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))):
    for i in net_worth: 
        Date += " {}".format(i)
#     print(Date)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file = open("logs/"+filename, 'a+')
    file.write(Date+"\n")
    file.close()

def get_csv_path():
    ret = []
    current_directory = os.getcwd()
    file_paths = [os.path.join(current_directory, file) for file in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, file))]

    for file_path in file_paths:
        if "csv" in file_path:
            ret.append(file_path)

    return ret