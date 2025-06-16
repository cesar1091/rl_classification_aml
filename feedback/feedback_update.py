import pandas as pd
import os

DATA_PATH = "data/processed/fake_aml_dataset_30_features.csv"
FEEDBACK_PATH = "data/feedback/new_feedback.csv"
UPDATED_PATH = "data/processed/updated_aml_dataset.csv"

def load_existing_dataset(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"❌ Could not find dataset at: {path}")

def load_feedback(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"❌ Could not find feedback at: {path}")

def update_labels_with_feedback(dataset, feedback):
    updated = dataset.copy()

    merged = updated.merge(feedback, on="client_id", how="left", suffixes=('', '_feedback'))

    # If feedback exists, overwrite the label_ros
    updated["label_ros"] = merged["label_ros_feedback"].combine_first(merged["label_ros"])

    # Drop temporary column if exists
    updated = updated.drop(columns=[col for col in updated.columns if col.endswith("_feedback")], errors="ignore")

    return updated

if __name__ == "__main__":
    print("🔄 Loading data...")
    dataset = load_existing_dataset(DATA_PATH)
    feedback = load_feedback(FEEDBACK_PATH)

    print(f"✅ Loaded {len(feedback)} feedback records.")

    updated_dataset = update_labels_with_feedback(dataset, feedback)

    updated_dataset.to_csv(UPDATED_PATH, index=False)
    print(f"✅ Updated dataset saved to: {UPDATED_PATH}")
