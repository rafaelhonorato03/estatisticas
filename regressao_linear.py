import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import stemgraphic
from io import StringIO
import scipy.stats as stats

# Script para o estudo de regressão linear

'''Problemas visuais e musculoesqueléticos associados ao uso de terminais de exibição visual (Visual Display Ter-minals – VDT) tornaram-se muito comuns nos últimos anos. Alguns pesquisadores concentraram-se na direção vertical do olhar como uma fonte de esforço e irritação ocular. Acredita-se que essa direção esteja intimamente relacionada com a Área da Superfície Ocular (ASO), de modo que é necessário um método para medi-la. Os da-dos representativos a seguir sobre y = ASO (cm2) e x = largura da fissura da pálpebra (isto é, a largura horizontal da abertura do olho, em cm) foram reproduzidos do artigo “Analysis of ocular surface area for comfortable VDT workstation layout” (Ergonomics, 1996: 877-884). A ordem na qual as observações foram obtidas não foi dada, de maneira que, por conveniência, são relacionadas na ordem crescente dos valores x.'''

x = [1.02, 1.21, 0.88, 0.98, 1.52, 1.83,
     1.50, 1.80, 1.74, 1.63, 2.00, 2.80,
     2.48, 2.47, 3.05, 3.18, 3.76, 3.68,
     3.82, 3.21, 4.27, 3.12, 3.99, 3.75,
     4.10, 4.18, 3.77, 4.34, 4.21, 4.92]

y = [0.40, 0.42, 0.48, 0.51, 0.57, 0.60,
     0.70, 0.75, 0.75, 0.78, 0.84, 0.95,
     0.99, 1.03, 1.12, 1.15, 1.20, 1.25,
     1.25, 1.28, 1.30, 1.34, 1.37, 1.40,
     1.43, 1.46, 1.49, 1.55, 1.58, 1.60] 

df = pd.DataFrame({'x':x, 'y':y})

sns.set_theme(style="whitegrid")
g = sns.JointGrid(data=df, x="x", y="y", marginal_ticks=True)

# Adicione o gráfico de dispersão (scatter) na área principal
g.plot_joint(sns.scatterplot, color="b")

# Adicione histogramas nas margens
g.plot_marginals(sns.histplot, color="b", element="step")
plt.show()


# Dados de amostra de água
# Dados
x = [7.01, 7.11, 7.12, 7.24, 7.94, 7.94, 8.04, 8.05, 8.07, 
     8.90, 8.94, 8.95, 8.97, 8.98, 9.85, 9.86, 9.86, 9.87]

y = [60, 67, 66, 52, 50, 45, 52, 48, 40,
     23, 20, 40, 31, 26, 9, 22, 13, 7]

# Criar DataFrame
df = pd.DataFrame({
    'pH': x,
    'Arsenico': y
})

sns.set_theme(style='whitegrid')

# Gráfico com regressão
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df, x='pH', y='Arsenico',
    scatter_kws={'s': 60, 'alpha': 0.7},
    line_kws={'color': 'red', 'linewidth': 2}
)

# Título e eixos
plt.title('Relação entre pH e Concentração de Arsênico', fontsize=16)
plt.xlabel('pH', fontsize=12)
plt.ylabel('Concentração de Arsênico', fontsize=12)
plt.show()

# Analise de Carros
import pandas as pd

# Dados
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

# Colunas
colunas = ["brand/model/year", "cap_vol", "consumo", "power", "weight", "cemm", "nu_cy", "Etype"]

# Criar DataFrame
df_carros = pd.DataFrame(dados, columns=colunas)

df_carros.info()
df_carros.describe()

stemgraphic.stem_graphic(df_carros['consumo'])
plt.show()

stemgraphic.stem_graphic(df_carros['cap_vol'])
plt.show()

stemgraphic.stem_graphic(df_carros['power'])
plt.show()

stemgraphic.stem_graphic(df_carros['weight'])
plt.show()

stemgraphic.stem_graphic(df_carros['cemm'])
plt.show()

stemgraphic.stem_graphic(df_carros['nu_cy'])
plt.show()

stemgraphic.stem_graphic(df_carros['Etype'])
plt.show()

sns.set_theme()
g = sns.lmplot(
    data=df_carros,
    x="cap_vol",
    y="consumo",
    hue="brand/model/year",
    height=5
)
plt.show()

sns.set_theme(style="whitegrid")
g = sns.JointGrid(data=df_carros,
                  x="cap_vol", y="consumo",
                  marginal_ticks=True)
g.plot_joint(sns.scatterplot, color="b")
g.plot_marginals(sns.rugplot, color="b")
plt.show()

sns.set_theme(style='ticks')
sns.pairplot(df_carros, hue='brand/model/year')
plt.show()

sns.set_theme(style="whitegrid")
palette = sns.color_palette("husl")
g = sns.relplot(
    data=df_carros,
    x="cap_vol", y="consumo",
    hue="brand/model/year", size="power",
    palette=palette, sizes=(10, 200),
)
sns.despine(left=True, bottom=True)
plt.show()

# Dados em formato CSV como string
dados_csv = """
brand/model/year,cap_vol,consumo,power,weight,cemm,classi
Gurgel BR800 0.8 1991,792,12.0,33,650,34.4,0
FIAT UNO Mille EP 1996,994,10.4,58,870,18.6,0
Hyundai HB20 Sense 2020,1000,12.8,80,989,14.5,0
FIAT Strada 1.4 2016,1368,10.3,86,1084,12.5,0
VolksWagen GOL 1.6 2015,1598,10.5,104,961,9.8,1
Chevrolet Cruze LTZ 1.8 2016,1796,8.5,144,1427,10.2,1
Honda Civic EXR 2016,1997,9.5,155,1294,10.9,1
Ford Focus 2.0 GLX 2012,1999,9.2,148,1347,10.4,1
BMW 325i 3.0 2012,2996,6.5,218,1460,7.1,1
AUDI A4 3.2 V6 Fsi 2011,3197,7.1,269,1610,6.4,1
Mercedes-Benz CLS 350 3.5 V6 2012,3498,6.6,306,1735,6.1,2
Mercedes-Benz CLS 500 5.5 V8 2007,5461,4.2,388,1760,5.4,2
Chevrolet Camaro SS 6.2 V8 2018,6162,6.4,461,1709,4.2,2
Pagani Zonda F 7.3 V12 2006,7291,3.0,602,1230,3.6,2
"""

df_carros_classificados = pd.read_csv(StringIO(dados_csv))

sns.lmplot(
    data=df_carros_classificados,
    x='cap_vol', y='power'
)
plt.show()

df_class_a = df_carros_classificados[df_carros_classificados['classi'] ==0]
df_class_b = df_carros_classificados[df_carros_classificados['classi'] ==1]
df_class_c = df_carros_classificados[df_carros_classificados['classi'] ==2]

plt.scatter(df_class_a['consumo'], df_class_a['weight'],
                       color='blue', marker= '*',
                       label='Classif0')
plt.scatter(df_class_b['consumo'], df_class_b['weight'],
                       color='red', marker= 'v',
                       label='Classif1')
plt.scatter(df_class_c['consumo'], df_class_c['weight'],
                       color='green', marker= '.',
                       label='Classif2')
plt.legend
plt.show()

plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
sns.lmplot(x='consumo',
           y='weight',
           hue='classi',
           data=df_carros_classificados)
plt.show()

sns.set_theme(style='ticks')
sns.pairplot(df_carros_classificados,
             hue='classi')
plt.show()


## Analisando o seguinte exercício:
'''O artigo "Alguma experiência de campo no uso de um método acelerado na estimativa da resistência do concreto em 28 dias" (Some field experience in the use of an accelerated method in estimating 28-day strength of concrete, J. of Amer. Concrete Institute, 1969: 895) fez a regressão de y = resistência do concreto usando tratamento padrão de 28 dias (1 psi = 7 KPa) em relação a x = resistência do concreto usando tratamento acelerado (psi).

Suponha que a equação da reta de regressão verdadeira seja  y=1800+1,3x .

a. Qual é o valor esperado da resistência aos 28 dias quando a resistência usando o tratamento acelerado é igual a 2500?

b. Até que ponto podemos esperar que a resistência aos 28 dias mude quando a resistência usando tratamento acelerado aumenta em 1 psi vezes?

c. Responda ao item (b) para um aumento de 100 psi.

d. Responda ao item (b) para uma diminuição de 100 psi.'''

intercept = 1800
slope = 1.3

# Faixa de X para regressão
x_vals = np.linspace(1900, 3100, 200)
y_vals = intercept + slope * x_vals

# Ponto de referência
x0 = 2500
y0 = intercept + slope * x0

# +100 e -100 psi
x_plus = x0 + 100
y_plus = intercept + slope * x_plus

x_minus = x0 - 100
y_minus = intercept + slope * x_minus

# --------------------------
# 2️⃣ Estética do gráfico
# --------------------------
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8), dpi=120)

# Linha de regressão
plt.plot(x_vals, y_vals, color="#1f77b4", linewidth=2.5, label="Equação de regressão")

# Ponto central
plt.scatter(x0, y0, color="#d62728", s=120, edgecolor='k', zorder=5)
plt.text(x0+20, y0+50, f'Ponto estimado\nx={x0}\ny={y0:.0f} psi',
         fontsize=12, fontweight='bold', color='#d62728')

# Pontos +100 e -100 psi
plt.scatter([x_plus, x_minus], [y_plus, y_minus],
            color="#2ca02c", marker='X', s=150, edgecolor='k', zorder=5)

# Anotações + setas
plt.annotate(f'+100 psi\nΔy = +130 psi',
             xy=(x_plus, y_plus), xytext=(x_plus+40, y_plus+60),
             fontsize=11, color="#2ca02c",
             arrowprops=dict(facecolor='#2ca02c', arrowstyle="->", linewidth=1.5))

plt.annotate(f'-100 psi\nΔy = -130 psi',
             xy=(x_minus, y_minus), xytext=(x_minus-200, y_minus-80),
             fontsize=11, color="#2ca02c",
             arrowprops=dict(facecolor='#2ca02c', arrowstyle="->", linewidth=1.5))

# --------------------------
# 3️⃣ Configuração dos eixos
# --------------------------
plt.title("📈 Regressão Linear: Resistência do Concreto (28 dias)", fontsize=14, pad=20)
plt.xlabel("Resistência (Tratamento Acelerado) [psi]", fontsize=10, labelpad=10)
plt.ylabel("Resistência (28 dias) [psi]", fontsize=10, labelpad=10)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.legend(fontsize=10)
sns.despine()
plt.tight_layout()
plt.show()

# Parâmetros
sigma = 350
slope = 1.3
intercept = 1800

# Itens a, b, c, d

# Item a: X = 2000
mu1 = intercept + slope * 2000
threshold = 5000
z1 = (threshold - mu1) / sigma
p_a = 1 - stats.norm.cdf(z1)

# Item b: X = 2500
mu2 = intercept + slope * 2500
z2 = (threshold - mu2) / sigma
p_b = 1 - stats.norm.cdf(z2)

# Item c: diferença
diff_mean = mu2 - mu1
diff_sigma = np.sqrt(2) * sigma
z3 = (1000 - diff_mean) / diff_sigma
p_c = 1 - stats.norm.cdf(z3)

# Item d: solve for delta X
delta_mu = stats.norm.ppf(0.95) * np.sqrt(2) * sigma
delta_X = delta_mu / slope

print(f"(a) P(Y > 5000 | X=2000) = {p_a:.4f}")
print(f"(b) P(Y > 5000 | X=2500) = {p_b:.4f}")
print(f"(c) P(Y2 - Y1 > 1000) = {p_c:.4f}")
print(f"(d) X2 deve exceder X1 em ≈ {delta_X:.1f} psi para P(Y2 > Y1)=0.95")

# Visualização com Seaborn
sns.set_theme(style="whitegrid")

x = np.linspace(4000, 6000, 500)

# Densidades
y1 = stats.norm.pdf(x, mu1, sigma)
y2 = stats.norm.pdf(x, mu2, sigma)

plt.figure(figsize=(12, 6))

# Curva para X=2000
plt.plot(x, y1, label=f'X=2000, μ={mu1:.0f}')
plt.fill_between(x, y1, where=(x > threshold), alpha=0.3, color='red',
                 label=f'Área > 5000: {p_a:.2%}')

# Curva para X=2500
plt.plot(x, y2, label=f'X=2500, μ={mu2:.0f}')
plt.fill_between(x, y2, where=(x > threshold), alpha=0.3, color='green',
                 label=f'Área > 5000: {p_b:.2%}')

# Linha limite
plt.axvline(threshold, color='black', linestyle='--')
plt.text(threshold+20, 0.0006, '5000 psi', fontsize=10)

plt.title('Probabilidade de Resistência do Concreto Ultrapassar 5000 psi')
plt.xlabel('Resistência (psi)')
plt.ylabel('Densidade')
plt.legend()
plt.show()


### Realização de Exercícios
'''A taxa de eficiência de uma amostra de aço imersa em um tanque de
fosfatação é o peso do revestimento de fosfato dividido pela perda do
metal (ambos em mg/pé2)( 1 mg/ft2 = 1,1 cg/m2). O artigo “Statistical
process control of a phosphate coating line”
(Wire J. Intl., maio 1997: 78-81) forneceu os dados a seguir sobre a
temperatura do tanque (x) e a taxa de eficiência (y).'''

# Dados
temp = [170,172, 173, 174, 174, 175, 176,
        177, 180, 180, 180, 180, 180, 181,
        181, 182, 182, 182, 182, 184, 184,
        185, 186, 188]

taxa = [0.84, 1.31, 1.42, 1.03, 1.07, 1.08, 1.04,
        1.80, 1.45, 1.60, 1.61, 2.13, 2.15, 0.84,
        1.43, 0.90, 1.81, 1.94, 2.68, 1.49, 2.52,
        3.0, 1.87, 3.08]

df_amostra_aco = pd.DataFrame({
    'Temp': temp,
    'Taxa': taxa
})

stemgraphic.stem_graphic(temp, scale=10)


f,ax = plt.subplots(figsize=(7,6))
sns.set_theme(style='ticks')
sns.boxplot(
    df_amostra_aco,
    x='Temp',
    )
sns.stripplot(df_amostra_aco,
              x='Temp',
              size= 4,
              color= '.3')
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.show()

sns.scatterplot(data=df_carros,
                x='cap_vol', y='consumo')
plt.show()

g = sns.JointGrid(data=df_carros, x="cap_vol", y="consumo", marginal_ticks=True)
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.histplot)
plt.show()