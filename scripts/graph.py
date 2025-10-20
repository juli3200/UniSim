# THIS FILE WAS GENERATED WITH GPT-5
import pandas as pd
import os
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(parent_dir, "..", "performance_results.csv")
df = pd.read_csv(csv_path)

# Plot
plt.figure(figsize=(8,5))
plt.plot(df["Ligands"], df["CPU(ms)"], marker="o", label="CPU (Optimized)")
plt.plot(df["Ligands"], df["CPU Unoptimized(ms)"], marker="o", label="CPU (Unoptimized)")
plt.plot(df["Ligands"], df["GPU(ms)"], marker="o", label="GPU")
plt.xscale("log")
plt.xlabel("Number of Ligands (log scale)")
plt.ylabel("Execution Time (ms)")
plt.title("CPU vs GPU Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()