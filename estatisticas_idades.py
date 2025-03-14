import pandas as pd
import math
import matplotlib.pyplot as plt

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
amplitude_classes = math.ceil(amplitude_idade / k)
print(f'Amplitude de classes: {amplitude_classes}')

# Somar a menor idade única com a amplitude de classe até atingir a idade máxima
idade_minima = df['Idade'].min()
idade_maxima = df['Idade'].max()

limites_classes = [idade_minima]
while limites_classes[-1] < idade_maxima:
    proximo_limite = limites_classes[-1] + amplitude_classes
    limites_classes.append(proximo_limite)

print(f'Limites das classes: {limites_classes}')

# Criar as classes e calcular a frequência
df['Classe'] = pd.cut(df['Idade'], bins=limites_classes, right=False)
frequencia = df.groupby('Classe')['Frequência'].sum()

# Criar um DataFrame com os limites das classes e suas frequências
frequencia_df = pd.DataFrame({
    'Limite Inferior': [interval.left for interval in frequencia.index],
    'Limite Superior': [interval.right for interval in frequencia.index],
    'Frequência': frequencia.values
})

# Calcular os pontos médios dos intervalos
frequencia_df['Ponto Médio'] = (frequencia_df['Limite Inferior'] + frequencia_df['Limite Superior']) / 2

print(frequencia_df)

# Gerar o gráfico
plt.figure(figsize=(10, 6))

# Histograma
plt.bar(
    x=[f"[{row['Limite Inferior']}, {row['Limite Superior']})" for _, row in frequencia_df.iterrows()],
    height=frequencia_df['Frequência'],
    color='skyblue',
    edgecolor='black',
    alpha=0.7,
    label='Histograma'
)

# Polígono de Frequência
plt.plot(
    frequencia_df['Ponto Médio'],
    frequencia_df['Frequência'],
    marker='o',
    color='red',
    label='Polígono de Frequência'
)

# Configurações do gráfico
plt.xlabel('Intervalos de Idade')
plt.ylabel('Frequência')
plt.title('Histograma e Polígono de Frequência')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()