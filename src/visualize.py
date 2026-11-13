import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Стиль для научных статей (Правило 3: Цветовая слепота)
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("colorblind")

df = pd.read_csv("data/processed/metrics_summary.csv")

# График 1: Влияние атаки (Epsilon) на F1-score
plt.figure(figsize=(8, 5))
sns.lineplot(
    data=df, x="epsilon", y="mean_f1", hue="model", 
    palette=palette, style="model", markers=True, dashes=False
)
plt.title("Impact of FGSM Attack on F1-Score")
plt.xlabel("Attack Strength (Epsilon)")
plt.ylabel("Mean F1-Score") # Правило 1: Явные оси
plt.tight_layout()
plt.savefig("reports/figures/fig1_attack_impact.pdf") # Векторный формат
plt.savefig("reports/figures/fig1_attack_impact.png", dpi=300) # PNG 300 DPI
plt.close()

# График 2: Распределение задержки (Latency)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df, x="model", y="p95_latency", hue="epsilon", palette="viridis"
)
plt.title("95th Percentile Latency (p95) by Model and Attack Strength")
plt.xlabel("Model")
plt.ylabel("Latency (ms)") # Правило 1: Явные оси
plt.tight_layout()
plt.savefig("reports/figures/fig2_latency.pdf")
plt.savefig("reports/figures/fig2_latency.png", dpi=300)
plt.close()
print("Plots generated.")
