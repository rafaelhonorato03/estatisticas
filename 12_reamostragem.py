import numpy as np
import matplotlib.pyplot as plt

# Amostra original (pequena para entendermos)
x = np.array([2, 4, 6, 8, 10])
n = len(x)

# --- JACKKNIFE ---
jackknife_means = []
for i in range(n):
    sample = np.delete(x, i)   # remove um elemento por vez
    jackknife_means.append(sample.mean())

print("Médias jackknife:", jackknife_means)
print("Média original:", x.mean())
print("Média das médias jackknife:", np.mean(jackknife_means))

# --- BOOTSTRAP ---
B = 1000  # número de reamostragens
bootstrap_means = []
for _ in range(B):
    sample = np.random.choice(x, size=n, replace=True)  # amostragem com reposição
    bootstrap_means.append(sample.mean())

print("\nBootstrap:")
print("Média bootstrap:", np.mean(bootstrap_means))
print("Intervalo de confiança 95%:",
      (np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5)))

# --- Visualização ---
plt.hist(bootstrap_means, bins=30, color="skyblue", edgecolor="black")
plt.axvline(x.mean(), color="red", linestyle="--", label="Média original")
plt.xlabel("Médias amostrais (bootstrap)")
plt.ylabel("Frequência")
plt.title("Distribuição Bootstrap das Médias")
plt.legend()
plt.show()
