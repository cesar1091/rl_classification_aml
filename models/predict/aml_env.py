import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class AmlEnv(gym.Env):
    def __init__(self, X=None, y=None):
        super().__init__()
        
        self.X = StandardScaler().fit_transform(X) if X is not None else None
        self.y = y
        self.index = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        self.index += 1
        done = self.index >= len(self.X)
        obs = self.X[self.index] if not done else np.zeros(self.X.shape[1])
        reward = 0.0  # reward is irrelevant in prediction
        return obs, reward, done, {}
