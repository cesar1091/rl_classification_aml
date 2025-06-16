import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
import mlflow

from aml_env import AmlEnv  # Importing the custom environment

MODEL_DIR = "models/"
MODEL_LOG_DIR = "logs/"
MODEL_NAME = "ppo_aml_agent"
ENV_PATH = "data/processed/fake_aml_dataset_30_features.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

def train_agent(env: gym.Env, timesteps: int = 100_000):
    check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=MODEL_DIR,
                                 log_path=MODEL_LOG_DIR, eval_freq=5000,
                                 deterministic=True, render=False)

    with mlflow.start_run(run_name="PPO_Agent_Training"):
        mlflow.log_param("algorithm", "PPO")
        mlflow.log_param("timesteps", timesteps)

        model.learn(total_timesteps=timesteps, callback=eval_callback)

        model.save(os.path.join(MODEL_DIR, MODEL_NAME))
        mlflow.log_artifact(os.path.join(MODEL_DIR, MODEL_NAME + ".zip"))

        print("✅ Model trained and saved.")
        return model

def load_agent(env: gym.Env):
    path = os.path.join(MODEL_DIR, MODEL_NAME + ".zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trained model not found at {path}")
    model = PPO.load(path, env=env)
    print("📦 Model loaded.")
    return model

def evaluate_agent(model, env, episodes=20):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward}")

    avg_reward = np.mean(rewards)
    print(f"🔍 Average reward over {episodes} episodes: {avg_reward:.2f}")
    mlflow.log_metric("avg_reward_eval", avg_reward)

if __name__ == "__main__":
    env = AmlEnv(ENV_PATH)

    # Training phase
    model = train_agent(env)

    # Evaluation phase
    evaluate_agent(model, env)
