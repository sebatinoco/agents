import argparse
from distutils.util import strtobool

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument('--env', type = str, default = 'MountainCarContinuous-v0', help = 'environment to run the experiment')
    parser.add_argument('--timesteps', type = int, default = 5000, help = 'timesteps of the experiment')
    parser.add_argument('--learning_start', type = int, default = 2000, help = 'timesteps of the experiment')
    parser.add_argument('--actor_lr', type = float, default = 1e-3, help = 'learning rate of the actor')
    parser.add_argument('--Q_lr', type = float, default = 1e-2, help = 'learning rate of the QNetwork')
    parser.add_argument('--Q_updates', type = int, default = 1, help = 'number of updates performed by critic')
    parser.add_argument('--gamma', type = float, default = .99, help = 'discount factor of the algorithm')
    parser.add_argument('--tau', type = float, default = .99, help = 'tau factor of the algorithm')
    parser.add_argument('--action_noise', type = float, default = 0.1, help = 'action noise of the agent')
    parser.add_argument('--buffer_size', type = int, default = 5000, help = 'size of replay buffer')
    parser.add_argument('--sample_size', type = int, default = 2000, help = 'batch size of the algorithm')
    parser.add_argument('--update_steps', type = int, default = 30, help = 'steps per update')
    parser.add_argument('--test_steps', type = int, default = 30, help = 'steps per test')
    parser.add_argument('--test_episodes', type = int, default = 30, help = 'episodes of test')
    parser.add_argument('--plot_steps', type = int, default = 1, help = 'steps per plot')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args