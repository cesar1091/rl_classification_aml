import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from aml_env import AmlEnv
import os

# 🔁 1. Definir entorno de predicción
class AmlPredictEnv(AmlEnv):
    def __init__(self, X):
        super().__init__(X=X, y=np.zeros(len(X)))  # y no se usa
        self.index = 0

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        self.index += 1
        done = self.index >= len(self.X)
        obs = self.X[self.index] if not done else np.zeros(self.X.shape[1])
        reward = 0.0
        return obs, reward, done, {}

# 🔄 2. Cargar datos reales
DATA_INPUT = "data/real_clients_to_predict.csv"
MODEL_PATH = "models/ppo_aml_agent.zip"
OUTPUT_PATH = "data/predictions/predictions.csv"

df = pd.read_csv(DATA_INPUT)
client_ids = df["client_id"].values
X = df.drop(columns=["client_id"]).values

# 🤖 3. Cargar modelo entrenado
model = PPO.load(MODEL_PATH)

# 🌐 4. Crear entorno y generar predicciones
env = AmlPredictEnv(X)
obs = env.reset()

predictions = []
while True:
    action, _ = model.predict(obs, deterministic=True)
    print(f"Predicción para cliente {client_ids[env.index]}: {'ROS' if action == 1 else 'No ROS'}")
    predictions.append(action)  # 0 o 1
    obs, _, done, _ = env.step(action)
    if done:
        break

# 💾 5. Guardar predicciones
df_output = pd.DataFrame({
    "client_id": client_ids,
    "prediction": predictions
})
df_output = pd.concat([df_output, df.drop(columns=["client_id"])], axis=1)

df_output.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predicciones guardadas en: {OUTPUT_PATH}")
