# -----------------------------
# 1️⃣ Importar bibliotecas
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

# -----------------------------
# 2️⃣ Criar os dados
# -----------------------------
# Velocidade do fluido (cm/s) e quantidade de gotículas (mg/m3)
x_vals = np.array([7, 10.3, 13.7, 16.6, 19.8, 22])
y_vals = np.array([479, 503, 487, 470, 458, 412])

# Transformar em DataFrame para manipulação fácil
df = pd.DataFrame({'x': x_vals, 'y': y_vals})

# -----------------------------
# 3️⃣ Regressão Linear
# -----------------------------
# Adicionar constante para o intercepto
X_linear = sm.add_constant(df['x'])
y = df['y']

# Ajustar modelo linear
model_linear = sm.OLS(y, X_linear).fit()
print("=== Regressão Linear ===")
print(model_linear.summary())

# Previsão
y_pred_linear = model_linear.predict(X_linear)

# -----------------------------
# 4️⃣ Gráfico da Regressão Linear
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(df['x'], df['y'], color='m', label='Dados Reais')
plt.plot(df['x'], y_pred_linear, color='b', label='Ajuste Linear')
plt.xlabel('Velocidade do fluxo do fluido (cm/s)')
plt.ylabel('Quantidade de gotículas de névoa (mg/m3)')
plt.title('Regressão Linear')
plt.legend()
plt.show()

# -----------------------------
# 5️⃣ Regressão Polinomial (grau 2)
# -----------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)  # sem coluna de 1, pois já usamos add_constant
X_poly = poly.fit_transform(df[['x']])

# Adicionar constante manualmente
X_poly = sm.add_constant(X_poly)

# Ajustar modelo polinomial
model_poly = sm.OLS(y, X_poly).fit()
print("=== Regressão Polinomial (grau 2) ===")
print(model_poly.summary())

# Previsão polinomial
y_pred_poly = model_poly.predict(X_poly)

# -----------------------------
# 6️⃣ Gráfico da Regressão Polinomial
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(df['x'], df['y'], color='m', label='Dados Reais')
# Para curva suave, gerar valores de x densos
x_smooth = np.linspace(df['x'].min(), df['x'].max(), 100)
X_smooth_poly = sm.add_constant(poly.transform(x_smooth.reshape(-1,1)))
y_smooth_poly = model_poly.predict(X_smooth_poly)
plt.plot(x_smooth, y_smooth_poly, color='g', label='Ajuste Polinomial')
plt.xlabel('Velocidade do fluxo do fluido (cm/s)')
plt.ylabel('Quantidade de gotículas de névoa (mg/m3)')
plt.title('Regressão Polinomial (grau 2)')
plt.legend()
plt.show()
