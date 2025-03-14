import pandas as pd

df = pd.read_excel(r'C:\Users\tabat\Documents\GitHub\estatisticas\Tabela de Idades.xlsx')
df = df.drop(df.index[-1])
df = df.astype(float)

print(df.head())

#Calculo da amplitude de idades
amplitude_idade = df['Idade'].max() - df['Idade'].min()
print(f'Amplitude de idades: {amplitude_idade}')

#Calculo do número de classes utilizando a regra da raiz quadrada
k = df['Idade'].count()
k = int(k ** 0.5)
print(f'Número de classes: {k}')

#Calculo da amplitude de classes
amplitude_classes = amplitude_idade / k
print(f'Amplitude de classes: {amplitude_classes}')

amplitude_idade = df['Idade'].max() - df['Idade'].min()
print(f'Amplitude de idades: {amplitude_idade}')
##amplitude_vida_feminina = df['Sum of Females  Life Expectancy'].max() - df['Sum of Females  Life Expectancy'].min()
#amplitude_vida_masculina = df['Sum of Males  Life Expectancy'].max() - df['Sum of Males  Life Expectancy'].min()
#amplitude_vida_geral = df['Sum of Life Expectancy  (both sexes)'].max() - df['Sum of Life Expectancy  (both sexes)'].min()