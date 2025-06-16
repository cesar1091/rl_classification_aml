import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

from aml_env import AmlEnv  # Importing the custom environment

MODEL_DIR = "models/"
MODEL_LOG_DIR = "logs/"
MODEL_NAME = "ppo_aml_agent"
ENV_PATH = "data/processed/fake_aml_dataset_30_features.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

def train_agent(env: gym.Env, timesteps: int = 100_000):
    check_env(env, warn=True)

    eval_env = AmlEnv(ENV_PATH, split="eval")  # aquí separamos para evaluación durante training
    model = PPO("MlpPolicy", env, verbose=1)

    eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR,
                                 log_path=MODEL_DIR, eval_freq=5000,
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
    y_true = []
    y_pred = []
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            y_true.append(info["true_label"])
            y_pred.append(action)

        rewards.append(total_reward)


        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward}")

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_reward = np.mean(rewards)

    print(f"📊 Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Avg Reward:{avg_reward:.2f}")
    with mlflow.start_run(run_name="PPO_Agent_Evaluation"):
        mlflow.log_metric("accuracy_eval", acc)
        mlflow.log_metric("precision_eval", prec)
        mlflow.log_metric("recall_eval", rec)
        mlflow.log_metric("f1_score_eval", f1)
        mlflow.log_metric("avg_reward_eval", avg_reward)

if __name__ == "__main__":
    train_env = AmlEnv(ENV_PATH, split="train")
    eval_env = AmlEnv(ENV_PATH, split="eval")

    # Entrenamiento
    model = train_agent(train_env)

    # Evaluación
    evaluate_agent(model, eval_env)
