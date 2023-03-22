#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
from typing import Callable
from IPython import display
import seaborn as sns
import time


# In[ ]:


def test_agent(env: gym.Env, policy: Callable, episodes: int = 10) -> None:
    
    for i in range(episodes):
        plt.figure()
        state = env.reset()
        state = state[0]
        done = False
        img = plt.imshow(env.render())
        while not done:
            p = policy(state)
            if isinstance(p, np.ndarray):
                action = np.random.choice(4, p=p)
            else:
                action = p
            next_state, _, done, _, _ = env.step(action)
            
            img.set_data(env.render())
            plt.axis('off')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            state = next_state
            time.sleep(1)


# In[ ]:


def plot_values(state_values, img):
    f, axes = plt.subplots(2, 1, figsize=(9,9))
    
    sns.heatmap(state_values, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                annot_kws={'weight': 'bold', 'size': 8}, 
                linewidths=1, 
                ax=axes[0])
    axes[1].imshow(img)
    axes[0].axis('off')
    axes[1].axis('off')
    f.subplots_adjust(hspace=0.3)
    plt.tight_layout()


# In[ ]:


def plot_policy(policy_probs, img):
    actions = {0:'U', 1:'R', 2:'D', 3:'L'}
    
    actions_from_prob = policy_probs.argmax(axis=-1)
    action_map = np.array(actions_from_prob.sha)

