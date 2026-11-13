import json
import time
import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# 1. Загрузка конфига
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# 2. Загрузка данных
df = pd.read_csv(config["data"]["sample_path"]).dropna()
X = df.select_dtypes(include=[np.number]).drop(columns=['Label'], errors='ignore')
y = df['Label']

# 3. Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 4. Цикл по моделям и атакам
log_file = "data/raw/test_run.jsonl"
with open(log_file, "w") as f:
    for model_name in ["xgboost", "random_forest"]:
        if model_name == "xgboost":
            model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        # Warm-up фаза (прогрев)
        for _ in range(5):
            model.fit(X_train, y_train)
        
        # Benchmark фаза
        for eps in [0.0, 0.01, 0.05, 0.1, 0.2]:
            start_time = time.perf_counter()
            
            # Симуляция атаки (добавляем шум)
            X_test_adv = X_test.copy().values
            if eps > 0:
                noise = np.random.uniform(-eps, eps, X_test_adv.shape)
                X_test_adv = X_test_adv + noise
            
            y_pred = model.predict(X_test_adv)
            f1 = f1_score(y_test, y_pred, average="weighted")
            acc = accuracy_score(y_test, y_pred)
            
            latency = time.perf_counter() - start_time
            
            # Запись в JSONL
            record = {
                "model": model_name,
                "epsilon": eps,
                "f1_score": f1,
                "accuracy": acc,
                "latency_ms": latency * 1000,
                "phase": "benchmark"
            }
            f.write(json.dumps(record) + "\n")

print(f"Done. Logs saved to {log_file}")
