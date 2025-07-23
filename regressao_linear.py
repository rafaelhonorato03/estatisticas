import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import stemgraphic

# Script para o estudo de regressão linear

'''Problemas visuais e musculoesqueléticos associados ao uso de terminais de exibição visual (Visual Display Ter-minals – VDT) tornaram-se muito comuns nos últimos anos. Alguns pesquisadores concentraram-se na direção vertical do olhar como uma fonte de esforço e irritação ocular. Acredita-se que essa direção esteja intimamente relacionada com a Área da Superfície Ocular (ASO), de modo que é necessário um método para medi-la. Os da-dos representativos a seguir sobre y = ASO (cm2) e x = largura da fissura da pálpebra (isto é, a largura horizontal da abertura do olho, em cm) foram reproduzidos do artigo “Analysis of ocular surface area for comfortable VDT workstation layout” (Ergonomics, 1996: 877-884). A ordem na qual as observações foram obtidas não foi dada, de maneira que, por conveniência, são relacionadas na ordem crescente dos valores x.'''

x = [0.40, 0.42, 0.51, 0.75, 0.48, 0.57, 0.60, 0.70, 0.75, 0.78, 
     0.84, 0.95, 0.99, 1.03, 1.12, 1.15, 1.20, 1.25, 1.25, 1.28,
     1.30, 1.34, 1.37, 1.40, 1.43, 1.46, 1.49, 1.55, 1.58, 1.60]

y = [1.02, 1.21, 0.98, 1.80, 0.88, 1.52, 1.83, 1.50, 1.74, 1.63,
     2.00, 2.80, 2.48, 2.47, 3.05, 3.18, 3.76, 3.68, 3.82, 3.21,
     4.27, 3.12, 3.99, 3.75, 4.10, 4.18, 3.77, 4.34, 4.21, 4.92]

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

df.info()
df.describe()

stemgraphic.stem_graphic(df_carros['consumo'])
plt.show()

sns.set_theme()
g = sns.lmplot(
    data=df_carros,
    x="consumo",
    y="power",
    hue="brand/model/year",
    height=5
)
plt.show()