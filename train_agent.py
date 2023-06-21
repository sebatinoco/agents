import gymnasium as gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer
from ddpg import DDPGAgent
from utils.test_agent import test_agent
from utils.plot_metrics import plot_metrics
from utils.parse_args import parse_args

args = parse_args()

env = gym.make(args['env'])

dim_states = env.observation_space.shape[0]
dim_actions = env.action_space.shape[0]

action_high = float(env.action_space.high)
action_low = float(env.action_space.low)

buffer = ReplayBuffer(dim_states, dim_actions, buffer_size = args['buffer_size'], sample_size = args['sample_size'])
agent = DDPGAgent(dim_states, dim_actions, action_high = action_high, action_low = action_low, 
              actor_lr = args['actor_lr'], Q_lr = args['Q_lr'], tau = args['tau'], Q_updates = args['Q_updates'], gamma = args['gamma'])

timesteps = args['timesteps']
learning_start = args['learning_start']
action_noise = args['action_noise']

update_steps = args['update_steps']
test_steps = args['test_steps']
test_episodes = args['test_episodes']
plot_steps = args['plot_steps']

avg_list = []
std_list = []

avg_reward, std_reward = test_agent(env, agent, test_episodes)
avg_list += [avg_reward]
std_list += [std_reward]

fig, axes = plt.subplots(1, 3, figsize = (20, 4))

ob_t, info = env.reset()
for t in tqdm(range(timesteps)):
    
    if t < learning_start:
        a_t = np.random.uniform(action_low, action_high, 1)
    else:
        a_t = agent.select_action(ob_t, action_noise = action_noise)
    
    ob_t1, r_t, terminated_t, truncated_t, info = env.step(a_t)
    done_t = terminated_t or truncated_t
    
    # modified reward
    #r_t = - (0.45 - ob_t1[0])
    r_t = 1 / (0.45 - ob_t1[0])
    
    experience = (ob_t, a_t, r_t, ob_t1, done_t)
    
    buffer.store_transition(*experience)
    
    if ((t % update_steps) == 0) & (t >= buffer.sample_size) & (t >= learning_start):
        agent.update(*buffer.sample())
        
        if t % plot_steps == 0:
            plot_metrics(fig, axes, avg_list, std_list, agent.actor_loss, agent.Q_loss, plot_steps)
    
    if t % test_steps == 0:
        avg_reward, std_reward = test_agent(env, agent, test_episodes)
        avg_list += [avg_reward]
        std_list += [std_reward]
    
    ob_t = ob_t1
    
    if done_t:
        ob_t, info = env.reset()