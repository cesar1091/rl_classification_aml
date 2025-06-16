import numpy as np
import pandas as pd

NUM_CLIENTS = 2000
ROS_RATIO = 0.2
MONTHS = 12
np.random.seed(42)

def generate_client_data(client_id, ros=False):
    base_tx_count = np.random.randint(30, 100)
    base_tx_amount = np.random.normal(1000, 300)
    base_cash_ratio = np.random.uniform(0.1, 0.6)
    base_intl_ratio = np.random.uniform(0.0, 0.2)
    base_weekday_ratio = np.random.uniform(0.7, 0.9)
    
    txs = []
    counterparties = set()
    prev_volume = []

    for month in range(MONTHS):
        # Behavioral drift for ROS
        drift = ros and month >= 6

        tx_count = np.random.poisson(base_tx_count * (1.5 if drift else 1.0))
        tx_amounts = np.random.normal(base_tx_amount * (2 if drift else 1), 300, tx_count)
        tx_amounts = np.clip(tx_amounts, 10, None)

        cash_flags = np.random.binomial(1, base_cash_ratio + (0.3 if drift else 0), tx_count)
        intl_flags = np.random.binomial(1, base_intl_ratio + (0.2 if drift else 0), tx_count)

        for amt, is_cash, is_intl in zip(tx_amounts, cash_flags, intl_flags):
            hour = np.random.choice(range(24))
            is_night = hour < 6 or hour > 22
            is_weekday = np.random.rand() < base_weekday_ratio

            cp_id = np.random.randint(10000, 99999)
            counterparties.add(cp_id)

            txs.append({
                "amount": amt,
                "is_cash": is_cash,
                "is_intl": is_intl,
                "hour": hour,
                "is_night": is_night,
                "is_weekday": is_weekday,
                "cp_id": cp_id,
                "month": month
            })

    df = pd.DataFrame(txs)

    # Feature engineering
    total_volume = df['amount'].sum()
    cash_volume = df[df['is_cash'] == 1]['amount'].sum()
    intl_volume = df[df['is_intl'] == 1]['amount'].sum()

    monthly_volumes = df.groupby('month')['amount'].sum().values
    post_spike_decline = int(len(monthly_volumes) > 7 and monthly_volumes[6] > 1.5 * np.mean(monthly_volumes[:6]) and np.mean(monthly_volumes[7:]) < monthly_volumes[6])

    features = {
        "client_id": client_id,
        "label_ros": int(ros),
        "total_tx_count": len(df),
        "avg_tx_amount": df['amount'].mean(),
        "tx_variance": df['amount'].var(),
        "cash_ratio": cash_volume / total_volume,
        "intl_ratio": intl_volume / total_volume,
        "cash_tx_count": df['is_cash'].sum(),
        "intl_tx_count": df['is_intl'].sum(),
        "avg_cash_amount": df[df['is_cash'] == 1]['amount'].mean() if df['is_cash'].sum() > 0 else 0,
        "intl_avg_amount": df[df['is_intl'] == 1]['amount'].mean() if df['is_intl'].sum() > 0 else 0,
        "intl_dest_diversity": df[df['is_intl'] == 1]['cp_id'].nunique(),
        "night_tx_ratio": df['is_night'].mean(),
        "weekday_tx_ratio": df['is_weekday'].mean(),
        "burstiness": df.groupby('month').size().var(),
        "avg_tx_interval": MONTHS / df.shape[0],
        "sudden_spike": int(monthly_volumes.max() > 2.5 * np.mean(monthly_volumes[:6])),
        "post_spike_decline": post_spike_decline,
        "volume_zscore_max": (monthly_volumes.max() - monthly_volumes.mean()) / monthly_volumes.std(),
        "monthly_volume_trend": np.polyfit(range(MONTHS), monthly_volumes, 1)[0],
        "counterparties_unique": len(counterparties),
        "avg_counterparty_freq": df.shape[0] / len(counterparties),
        "new_counterparties_pct": np.mean([np.random.rand() < (0.3 if ros else 0.05) for _ in range(len(counterparties))]),
        "repeated_tx_amounts": int((df['amount'].value_counts().max() / df.shape[0]) > 0.1),
        "structuring_score": int(df[(df['amount'] > 9000) & (df['amount'] < 10000)].shape[0] > 3),
        "volume_skewness": df['amount'].skew(),
        "known_alert_history": int(ros and np.random.rand() < 0.5),
        "high_risk_country_flag": int(ros and np.random.rand() < 0.4),
        "low_activity_followed_by_spike": int(np.mean(monthly_volumes[:6]) < 0.5 * monthly_volumes[6] if len(monthly_volumes) > 6 else 0),
        "tx_count_variance": df.groupby('month').size().var()
    }

    return features

def generate_dataset():
    rows = [generate_client_data(i, ros=(i < NUM_CLIENTS * ROS_RATIO)) for i in range(NUM_CLIENTS)]
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/processed/fake_aml_dataset_30_features.csv", index=False)
    print(f"✅ Dataset with {df.shape[0]} clients and {df.shape[1]} features generated.")
