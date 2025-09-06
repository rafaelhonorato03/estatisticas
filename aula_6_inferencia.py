import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  # estilo mais agradável para gráficos

# ===============================
# 1️⃣ Dados: casas e microdureza
# ===============================

# Dados fictícios: preço de casa depende de tamanho, idade e quartos
df_casas = pd.DataFrame({
    "preco": [200, 250, 300, 400, 500, 600],
    "tamanho": [50, 60, 70, 80, 90, 100],
    "idade": [30, 25, 20, 15, 10, 5],
    "quartos": [2, 2, 3, 3, 4, 4]
})

# Dados de microdureza
microdureza = np.array([6.1,7.5,6.2,7.6,8.6,6.1,6.4,6.7,6.9,7.2,
                        6.1,6.2,6.1,6.1,7.5,7.6,7.2,7.2])
N = np.array([10, 15, 20, 25, 30, 10, 15, 20, 25, 30, 
              10, 15, 20, 25, 30, 35, 40, 45])
F = np.array([1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2])
df_micro = pd.DataFrame({'Y': microdureza, 'N': N, 'F': F})

# ===============================
# 2️⃣ Função para ajustar modelo
# ===============================
def ajustar_modelo(df, y_col, X_cols):
    X = sm.add_constant(df[X_cols])
    y = df[y_col]
    modelo = sm.OLS(y, X).fit()
    return modelo

# Modelo preço de casas
modelo_casas = ajustar_modelo(df_casas, "preco", ["tamanho","idade","quartos"])
print("Resumo do modelo de preço de casas:\n")
print(modelo_casas.summary())

# Modelo microdureza
modelo_micro = ajustar_modelo(df_micro, "Y", ["N","F"])
print("\nResumo do modelo de microdureza:\n")
print(modelo_micro.summary())

# ===============================
# 3️⃣ Predição para novos valores
# ===============================
def predizer(modelo, novo_df, alpha=0.05):
    novo_df = sm.add_constant(novo_df, has_constant='add')  # garante intercepto
    pred = modelo.get_prediction(novo_df)
    return pred.summary_frame(alpha=alpha)

novo_ponto = pd.DataFrame({'N':[20], 'F':[1]})
pred_resumo = predizer(modelo_micro, novo_ponto)
print("\nPredição para N=20, F=1:")
print(pred_resumo)

# ===============================
# 4️⃣ Gráficos de diagnóstico
# ===============================
def diagnostico(modelo):
    residuos = modelo.resid
    residuos_pad = residuos / np.std(residuos)

    # Resíduos padronizados vs preditos
    plt.figure(figsize=(8,5))
    plt.scatter(modelo.fittedvalues, residuos_pad, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Valores Preditos")
    plt.ylabel("Resíduos Padronizados")
    plt.title("Resíduos Padronizados vs Valores Preditos")
    plt.show()

    # QQ-Plot dos resíduos
    plt.figure(figsize=(6,6))
    sm.qqplot(residuos, line='45', fit=True)
    plt.title("QQ-Plot dos Resíduos")
    plt.show()

diagnostico(modelo_micro)
