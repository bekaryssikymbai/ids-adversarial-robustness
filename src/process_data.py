import json
import pandas as pd
import numpy as np

# Загрузка сырых логов
data = []
with open("data/raw/test_run.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Группировка и расчет статистики
summary = df.groupby(['model', 'epsilon']).agg(
    mean_f1=('f1_score', 'mean'),
    std_f1=('f1_score', 'std'),
    p95_latency=('latency_ms', lambda x: np.percentile(x, 95)),
    p99_latency=('latency_ms', lambda x: np.percentile(x, 99))
).reset_index()

# Сохранение агрегата
summary.to_csv("data/processed/metrics_summary.csv", index=False)
print("Processed data saved.")
