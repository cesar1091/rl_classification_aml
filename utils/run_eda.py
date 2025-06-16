import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# Set up plot output directory
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "data/processed/fake_aml_dataset_30_features.csv"
df = pd.read_csv(DATA_PATH)

# Split classes
df_ros = df[df["label_ros"] == 1]
df_noros = df[df["label_ros"] == 0]

# ========== 1. Class Distribution ==========
plt.figure()
df["label_ros"].value_counts().plot(kind="bar", title="Class Distribution")
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")

# ========== 2. Mean comparison ==========
mean_comparison = pd.concat([
    df_ros.describe().T["mean"],
    df_noros.describe().T["mean"]
], axis=1)
mean_comparison.columns = ["ROS_mean", "No_ROS_mean"]
mean_comparison["difference"] = mean_comparison["ROS_mean"] - mean_comparison["No_ROS_mean"]
mean_comparison = mean_comparison.sort_values("difference", key=abs, ascending=False)

mean_comparison.to_csv(f"{OUTPUT_DIR}/mean_comparison.csv")
top_features = mean_comparison.head(6).index.tolist()

# ========== 3. Boxplots ==========
for col in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="label_ros", y=col, data=df)
    plt.title(f"Boxplot: {col}")
    plt.savefig(f"{OUTPUT_DIR}/boxplot_{col}.png")
    plt.close()

# ========== 4. KDE Distributions ==========
for col in top_features:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(df_ros[col], label="ROS", fill=True, common_norm=False)
    sns.kdeplot(df_noros[col], label="No ROS", fill=True, common_norm=False)
    plt.title(f"Distribution: {col}")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/kde_{col}.png")
    plt.close()

# ========== 5. Correlation Matrices ==========
plt.figure(figsize=(10, 8))
sns.heatmap(df_ros.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Correlation Matrix: ROS Clients")
plt.savefig(f"{OUTPUT_DIR}/corr_matrix_ros.png")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df_noros.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Correlation Matrix: No ROS Clients")
plt.savefig(f"{OUTPUT_DIR}/corr_matrix_noros.png")
plt.close()

# ========== 6. MLflow Logging ==========
with mlflow.start_run(run_name="EDA_ROS"):
    mlflow.log_param("dataset", DATA_PATH)
    mlflow.log_artifact(f"{OUTPUT_DIR}/class_distribution.png")
    mlflow.log_artifact(f"{OUTPUT_DIR}/mean_comparison.csv")

    for col in top_features:
        mlflow.log_artifact(f"{OUTPUT_DIR}/boxplot_{col}.png")
        mlflow.log_artifact(f"{OUTPUT_DIR}/kde_{col}.png")

    mlflow.log_artifact(f"{OUTPUT_DIR}/corr_matrix_ros.png")
    mlflow.log_artifact(f"{OUTPUT_DIR}/corr_matrix_noros.png")

print("✅ EDA completed. Outputs saved in 'eda_outputs/' and logged to MLflow.")
