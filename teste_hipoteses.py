# Dados cientificos
# x = velocidade do fluxo do fluido para um óleo 5% solúvel (cm/s)
# y = quantidade de gotículas de névoa com diâmetro menor que 10 mm (mg/m3)

# Perguntas: Regressão com dispersão linear
# 2 - Regressão linear simples atende?
# 3 - Proporção da variação no volume de névoa atribuída à relação de regressão linear simples entre velocidade e névoa
# 4 - Aumentar a velocidade de 100 para 1000. Quanto de x aumenta? Há evidência de que o aumento médio verdadeiro em y seja menor que 0,6?
# 5 - Mudança média verdadeira na névoa associada com um aumento da velocidade de 1 cm/s

import pandas as pd
from tabulate import tabulate
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import t

table = [["x", 89, 177, 189, 354, 362, 442, 965],
          ["y", 0.40, 0.60, 0.48, 0.66, 0.61, 0.69, 0.99]]

print(tabulate(table))

df = pd.DataFrame({"x": table[0][1:], "y": table[1][1:]})
print(df)

sns.lmplot(
    data=df,
    x="x",
    y="y",
)
plt.xlabel('Velocidade do fluxo do fluido (cm/s)')
plt.ylabel('Quantidade de gotículas de névoa (mg/m3)')
plt.title('Gráfico de Dispersão')
plt.show()

# Analises do stats
x = sm.add_constant(df['x'])
modelo = sm.OLS(df['y'],df['x']).fit()
print(modelo.summary())

# Buscar tabela tstudent
alpha = 0.05 # Nível de significancia = 5%
df = len(df['x']) - 2 # Graus de liberdade (Número de amostra menos dois)
v = t.ppf(1 - alpha/2, df)
tcrit = v
print(f'tcrit= : {v}')

