import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Dados originais
lstx2 = (100, 125, 125, 150, 150, 200, 200, 250, 250, 300, 300, 350, 400, 400)
lsty2 = (150, 140, 180, 210, 190, 320, 280, 400, 430, 440, 390, 600, 610, 670)

# Ajustar regressão linear (reta: y = ax + b)
coef = np.polyfit(lstx2, lsty2, 1)
a, b = coef
print(f"Equação da reta ajustada: y = {a:.2f}x + {b:.2f}")

# Predição
y_pred = a * np.array(lstx2) + b

# Resíduos
resi = np.array(lsty2) - y_pred

# Resíduos padronizados (z-score)
standardized_residuals = (resi - np.mean(resi)) / np.std(resi)

# Criar DataFrame completo
df = pd.DataFrame({
    "x": lstx2,
    "y": lsty2,
    "y_pred": y_pred,
    "residuals": resi,
    "standardized_residuals": standardized_residuals
})

# Correlação
corr = df[["x", "y"]].corr().iloc[0,1]
print(f'Correlação entre x e y: {corr:.4f}')

# Gráfico de dispersão com reta de regressão
plt.figure(figsize=(8,6))
sns.scatterplot(x="x", y="y", data=df, label="Dados reais")
sns.lineplot(x="x", y="y_pred", data=df, color="red", label="Regressão Linear")
plt.title("Dispersão entre x e y com regressão linear")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de calor da correlação")
plt.show()

# Estatísticas
print("\n--- INFO ---")
print(df.info())
print("\n--- DESCRIBE ---")
print(df.describe())
print("\n--- HEAD ---")
print(df.head())
print("\n--- CORRELAÇÃO COMPLETA ---")
print(df.corr())


# -------------------------
# Base de dados
# -------------------------
dados = [
    ["Gurgel BR800 0.8 1991", 792, 12, 33, 650, 34.4, 2, 0],
    ["FIAT UNO Mille EP 1996", 994, 10.4, 58, 870, 18.6, 4, 0],
    ["Hyundai HB20 Sense 2020", 1000, 12.8, 80, 989, 14.5, 3, 0],
    ["FIAT Strada 1.4 2016", 1368, 10.3, 86, 1084, 12.5, 4, 0],
    ["VolksWagen GOL 1.6 2015", 1598, 10.5, 104, 961, 9.8, 4, 0],
    ["Chevrolet Cruze LTZ 1.8 2016", 1796, 8.5, 144, 1427, 10.2, 4, 0],
    ["Honda Civic EXR 2016", 1997, 9.5, 155, 1294, 10.9, 4, 0],
    ["Ford Focus 2.0 GLX 2012", 1999, 9.2, 148, 1347, 10.4, 4, 0],
    ["BMW 325i 3.0 2012", 2996, 6.5, 218, 1460, 7.1, 6, 0],
    ["AUDI A4 3.2 V6 Fsi 2011", 3197, 7.1, 269, 1610, 6.4, 6, 0],
    ["Mercedes-Benz CLS 350 3.5 V6 2012", 3498, 6.6, 306, 1735, 6.1, 6, 0],
    ["Mercedes-Benz CLS 500 5.5 V8 2007", 5461, 4.2, 388, 1760, 5.4, 8, 0],
    ["Chevrolet Camaro SS 6.2 V8 2018", 6162, 6.4, 461, 1709, 4.2, 8, 0],
    ["Pagani Zonda F 7.3 V12 2006", 7291, 3, 602, 1230, 3.6, 12, 0],
]

colunas = ["brand/model/year", "cap_vol", "consumo", "power", "weight", "cemm", "nu_cy", "Etype"]
df_carros = pd.DataFrame(dados, columns=colunas)

# -------------------------
# Modelo de regressão linear
# -------------------------
X = df_carros["cap_vol"]
y = df_carros["consumo"]

# Adicionando constante (intercepto)
X_const = sm.add_constant(X)

# Ajuste do modelo
modelo = sm.OLS(y, X_const).fit()

# Predições e resíduos
df_carros["y_pred"] = modelo.predict(X_const)
df_carros["residuals"] = y - df_carros["y_pred"]
df_carros["std_resid"] = modelo.get_influence().resid_studentized_internal

# -------------------------
# VISUALIZAÇÕES
# -------------------------

# 1. Dispersão com reta de regressão
plt.figure(figsize=(8,6))
sns.scatterplot(x="cap_vol", y="consumo", data=df_carros, s=80, label="Dados reais")
sns.lineplot(x="cap_vol", y="y_pred", data=df_carros, color="red", label="Regressão Linear")
plt.title("Dispersão com reta de regressão")
plt.show()

# 2. Resíduos vs valores ajustados
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_carros["y_pred"], y=df_carros["residuals"])
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Valores ajustados (ŷ)")
plt.ylabel("Resíduos (e)")
plt.title("Resíduos vs Valores Ajustados")
plt.show()

# 3. Resíduos padronizados
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_carros["y_pred"], y=df_carros["std_resid"])
plt.axhline(0, color="red", linestyle="--")
plt.axhline(2, color="orange", linestyle="--", label="Limite superior")
plt.axhline(-2, color="orange", linestyle="--", label="Limite inferior")
plt.xlabel("Valores ajustados (ŷ)")
plt.ylabel("Resíduos padronizados (e*)")
plt.title("Resíduos Padronizados vs Valores Ajustados")
plt.legend()
plt.show()

# 4. Histograma dos resíduos
plt.figure(figsize=(8,6))
sns.histplot(df_carros["residuals"], bins=10, kde=True)
plt.title("Distribuição dos Resíduos")
plt.show()

# 5. QQ-Plot (normalidade dos resíduos)
sm.qqplot(df_carros["residuals"], line='45')
plt.title("QQ-Plot dos Resíduos")
plt.show()

# 6. Heatmap de correlação
plt.figure(figsize=(8,6))
sns.heatmap(df_carros.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor das Correlações")
plt.show()

# 7. Pairplot (relações entre todas as variáveis numéricas)
sns.pairplot(df_carros[["cap_vol", "consumo", "power", "weight", "cemm"]], diag_kind="kde")
plt.suptitle("Relações Multivariadas", y=1.02)
plt.show()

# 8. Gráfico de barras: consumo por modelo
plt.figure(figsize=(10,6))
sns.barplot(y="brand/model/year", x="consumo", data=df_carros, palette="viridis")
plt.title("Consumo (km/L) por Modelo de Carro")
plt.xlabel("Consumo (km/L)")
plt.ylabel("Modelo")
plt.show()

# 9. Gráfico de dispersão 3D (cap_vol, power, consumo)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df_carros["cap_vol"], df_carros["power"], df_carros["consumo"], c=df_carros["consumo"], cmap="viridis", s=80)
ax.set_xlabel("Capacidade Volumétrica")
ax.set_ylabel("Potência (cv)")
ax.set_zlabel("Consumo (km/L)")
plt.title("Dispersão 3D: Cap_Vol x Potência x Consumo")
plt.show()
