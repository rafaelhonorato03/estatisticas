import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\tabat\Documents\GitHub\estatisticas\Tabela de Idades.xlsx')
df = df.drop(df.index[-1])
df = df.astype(float)

# Calculo da amplitude de idades
amplitude_idade = df['Idade'].max() - df['Idade'].min()
k = int(df['Idade'].count() ** 0.5)
amplitude_classes = math.ceil(amplitude_idade / k)

# Criar os limites das classes
idade_minima = df['Idade'].min()
idade_maxima = df['Idade'].max()
limites_classes = [idade_minima]
while limites_classes[-1] < idade_maxima:
    limites_classes.append(limites_classes[-1] + amplitude_classes)

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

# Calcular a frequência acumulada
frequencia_df['Frequência Acumulada'] = frequencia_df['Frequência'].cumsum()

print(frequencia_df)

# Gráfico 1: Histograma
plt.figure(figsize=(10, 6))
plt.bar(
    x=[f"[{row['Limite Inferior']}, {row['Limite Superior']})" for _, row in frequencia_df.iterrows()],
    height=frequencia_df['Frequência'],
    color='skyblue',
    edgecolor='black',
    alpha=0.7
)
plt.xlabel('Intervalos de Idade')
plt.ylabel('Frequência')
plt.title('Histograma de Frequências')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Gráfico 2: Polígono de Frequência
plt.figure(figsize=(10, 6))
plt.plot(
    frequencia_df['Ponto Médio'],
    frequencia_df['Frequência'],
    marker='o',
    color='red',
    label='Polígono de Frequência'
)
plt.xlabel('Pontos Médios dos Intervalos')
plt.ylabel('Frequência')
plt.title('Polígono de Frequência')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Gráfico 3: Ogiva (Frequência Acumulada)
plt.figure(figsize=(10, 6))
plt.plot(
    frequencia_df['Limite Superior'],
    frequencia_df['Frequência Acumulada'],
    marker='o',
    color='green',
    label='Ogiva'
)
plt.xlabel('Limites Superiores dos Intervalos')
plt.ylabel('Frequência Acumulada')
plt.title('Ogiva (Frequência Acumulada)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()