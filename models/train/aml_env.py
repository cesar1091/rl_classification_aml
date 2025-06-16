import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class AmlEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, csv_path, split="train", test_size=0.2, random_state=42):
        super(AmlEnv, self).__init__()

        full_data = pd.read_csv(csv_path)
        full_data = shuffle(full_data, random_state=random_state).reset_index(drop=True)

        train_data, test_data = train_test_split(full_data, test_size=test_size, random_state=random_state, stratify=full_data["label_ros"])

        if split == "train":
            self.data = train_data
        elif split == "eval":
            self.data = test_data
        else:
            raise ValueError("split must be 'train' or 'eval'")

        self.features = self.data.drop(columns=["client_id", "label_ros"]).values
        self.labels = self.data["label_ros"].values

        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]
        self.current_index = 0

        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = np.random.randint(0, self.n_samples)
        obs = self.features[self.current_index]
        return obs.astype(np.float32), {}

    def step(self, action):
        label = self.labels[self.current_index]

        reward = 1.0 if action == label else -1.0

        terminated = True
        truncated = False
        obs = self.features[self.current_index]
        info = {
            "true_label": label,
            "predicted": action
        }

        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        print(f"Client {self.current_index}: true={self.labels[self.current_index]}")

    def close(self):
        pass
