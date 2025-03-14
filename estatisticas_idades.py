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