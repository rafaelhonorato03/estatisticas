import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import scipy.stats
from scipy import stats
from statsmodels.graphics.gofplots import ProbPlot
from io import StringIO

df_carros = pd.read_excel(r'modelagem_inferencia_estatistica\carros.xlsx')
print(df_carros)
print(df_carros.info())
print(df_carros.describe())

df_carros = df_carros[df_carros['Etype'] == 0] #0 = automático, 1 = manual

regmul = smf.ols('consumo ~ power + weight + nu_cy', data=df_carros) #Regressão Multipla
res =  regmul.fit() #Ajuste do modelo
print(res.summary()) #Resumo do modelo

f = res.fvalue #F-value
k = res.df_model #Graus de liberdade do modelo
n = res.nobs #Número de observações
dfn = k #Graus de liberdade do numerador
dfd = n - (k + 1) #Graus de liberdade do denominador
alpha = 0.1 #Nível de confianca
f_critico = scipy.stats.f.ppf(1 - alpha, dfn, dfd)
print(f'F-critico: {f_critico}') #F-critico

y_prev = list(res.predict()) #Valores ajustados
resi = list(res.resid) #Resíduos
influence = res.get_influence() #Influência dos pontos
std_resid = list(influence.resid_studentized_internal) #Resíduos padronizados
prop = np.divide(resi, std_resid) #Proporção

y = list(df_carros['consumo'])
dftab = pd.DataFrame(list(zip(y, y_prev, resi, std_resid, prop)),
                     columns=['y', 'y_p', 'e', 'e*', 'e/e*'])
print(dftab.head(10))

sns.lmplot(x='y', y='y_p', data=dftab)
plt.xlabel('Valores observados (y)= Consumo (km/L)')
plt.ylabel('Valores esperados (ŷ)= Consumo (km/L)')
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.show()

sns.scatterplot(x='y', y='e', data=dftab)
plt.xlabel('Valores observados (y)= Consumo (km/L)')
plt.ylabel('Resíduos padronizados (e*)')
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.show()

QQ = ProbPlot(influence.resid_studentized_internal)
plot_lm = QQ.qqplot(line='45', alpha=0.5, color='red', lw=1)
plot_lm.axes[0].set_xlabel('Percentil')
plot_lm.axes[0].set_ylabel('Resíduos padronizados')
plt.show()

#Analise de amostras influentes
infl = res.get_influence()
print(infl.hat_matrix_diag)

#Valores dos resíduos
leviers = infl.hat_matrix_diag #leverage
sigma_err = np.sqrt(res.scale) #regression standard error
res_stds = std_resid/(sigma_err*np.sqrt(1.0-leviers))
print(res_stds)

print(infl.summary_frame().filter(['hat_diag', 'student_resid', 'dffits', 'cooks_d']))

corr = dftab.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1)
plt.show()

print(corr)

mc2 = corr**2
print(mc2)

vif = np.linalg.inv(mc2)
print(vif)
