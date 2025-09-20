# SEÇÃO 1: REGRESSÃO LINEAR MÚLTIPLA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import scipy.stats as stats
from statsmodels.graphics.gofplots import ProbPlot

# Dados para o modelo de regressão linear
dx1 = (0.93, 1.11, 0.93, 1.11, 0.93, 1.11, 0.93, 1.11, 1.02, 1.02, 1.02, 1.02)
dx2 = (1.00, 1.00, 1.00, 1.00, 1.40, 1.40, 1.40, 1.40, 1.18, 1.18, 1.18, 1.18)
dx3 = (0.20, 0.20, 0.50, 0.50, 0.20, 0.20, 0.50, 0.50, 0.31, 0.31, 0.31, 0.31)
dy = (32.95, 38.72, 35.20, 38.72, 32.27, 39.71, 33.67, 38.72, 35.20, 33.67, 36.02, 32.27)

# DataFrame original
df_linear = pd.DataFrame({
    "x1": dx1,
    "x2": dx2,
    "x3": dx3,
    "y": dy
})

# Transformação logarítmica para linearizar a relação
df_linear['lnx1'] = np.log(df_linear['x1'])
df_linear['lnx2'] = np.log(df_linear['x2'])
df_linear['lnx3'] = np.log(df_linear['x3'])
df_linear['lny'] = np.log(df_linear['y'])

print("Primeiras linhas do DataFrame (Regressão Linear):")
print(df_linear.head())

# Ajustar o modelo de regressão linear múltipla
modelo_linear = smf.ols('lny ~ lnx1 + lnx2 + lnx3', data=df_linear)
res_linear = modelo_linear.fit()

# Exibir o resumo detalhado do modelo
print("\n" + "="*80)
print("RESUMO DO MODELO DE REGRESSÃO LINEAR MÚLTIPLA")
print("="*80)
print(res_linear.summary())


# SEÇÃO 2: DIAGNÓSTICO DO MODELO DE REGRESSÃO LINEAR

print("\n" + "="*80)
print("DIAGNÓSTICO DO MODELO DE REGRESSÃO LINEAR")
print("="*80)

# Gráficos de diagnóstico para verificar suposições
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico de Normalidade (Q-Q Plot) para os resíduos
# Avalia se os resíduos seguem uma distribuição normal
qqplot = ProbPlot(res_linear.get_influence().resid_studentized_internal)
qqplot.qqplot(line='45', ax=ax1, color='#1F77B4')
ax1.set_title('Q-Q Plot dos Resíduos (Normalidade)')
ax1.grid(True)
ax1.set_xlabel('Quantis Teóricos')
ax1.set_ylabel('Resíduos Padronizados')

# Gráfico de Resíduos vs. Valores Ajustados (Homoscedasticidade)
# Avalia se a variância dos resíduos é constante
sns.residplot(x=res_linear.fittedvalues, y=res_linear.resid, ax=ax2,
              lowess=True, line_kws={'color': 'red', 'lw': 2, 'alpha': 0.8},
              scatter_kws={'alpha': 0.6})
ax2.set_title('Resíduos vs. Valores Ajustados (Homoscedasticidade)')
ax2.set_xlabel('Valores Ajustados')
ax2.set_ylabel('Resíduos')
ax2.grid(True)
plt.tight_layout()
plt.show()

# SEÇÃO 3: ANÁLISE DE MULTICOLINEARIDADE (REGRESSÃO LINEAR)

print("\n" + "="*80)
print("ANÁLISE DE MULTICOLINEARIDADE")
print("="*80)

# Calcula a matriz de correlação
corr_matrix_linear = df_linear[['lnx1', 'lnx2', 'lnx3', 'lny']].corr()

# Visualiza a matriz de correlação em um heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_linear, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação das Variáveis')
plt.show()

# SEÇÃO 4: REGRESSÃO LOGÍSTICA

# Dados para o modelo de regressão logística
lstx1 = (1.80, 1.65, 2.70, 3.67, 1.41, 1.76, 2.10, 2.10, 4.57, 3.59,
         8.33, 2.86, 2.58, 2.90, 3.89, 0.80, 0.60, 1.30, 0.83, 0.57,
         1.44, 2.08, 1.50, 1.38, 0.94, 1.58, 1.67, 3.00, 2.21)
lstx2 = (2.40, 2.54, 0.84, 1.68, 2.41, 1.93, 1.77, 1.50, 2.43, 5.55,
         5.58, 2.00, 3.68, 1.13, 2.49, 1.37, 1.27, 0.87, 0.97, 0.94,
         1.00, 0.78, 1.03, 0.82, 1.30, 0.83, 1.05, 1.19, 0.86)
lsty = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0)

# Construir o DataFrame para a regressão logística
df_logistic = pd.DataFrame({
    "x1": lstx1,
    "x2": lstx2,
    "y": lsty
})

# Definir variáveis independentes (x) e dependente (y)
x_log = sm.add_constant(df_logistic[['x1', 'x2']])
y_log = df_logistic['y']

# Construir e ajustar o modelo de regressão logística
modelo_log = sm.Logit(y_log, x_log).fit()

print("\n" + "="*80)
print("RESUMO DO MODELO DE REGRESSÃO LOGÍSTICA")
print("="*80)
print(modelo_log.summary())

# Calcular e exibir a razão das chances (Odds Ratios)
odds_ratios = np.exp(modelo_log.params)
print("\nRazão das Chances (Odds Ratios):")
print(odds_ratios)

# Gerar previsões do modelo e a matriz de confusão
yhat_log = modelo_log.predict(x_log)
prediction_log = list(map(round, yhat_log))

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_log, prediction_log)
print("\nMatriz de Confusão:\n", cm)

accuracy = accuracy_score(y_log, prediction_log)
print(f"\nAcurácia do Modelo: {accuracy:.2f}")

# Análise de multicolinearidade para o modelo logístico
corr_matrix_log = df_logistic.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_log, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação (Regressão Logística)')
plt.show()