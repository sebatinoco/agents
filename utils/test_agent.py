import numpy as np

def test_agent(env, agent, test_episodes):
    
    reward_array = np.zeros(test_episodes)

    ob_t, info = env.reset()
    for episode in range(test_episodes):
        
        done = False
        truncated = False
        reward_sum = 0
        while not (done or truncated):
        
            action = agent.select_action(ob_t, action_noise = 0.0)
            
            ob_t, reward, done, truncated, info = env.step(action)
            
            reward_sum += reward
        
        reward_array[episode] = reward_sum
        
    return np.mean(reward_array), np.std(reward_array)