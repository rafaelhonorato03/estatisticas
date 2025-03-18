import pandas as pd
import math
import matplotlib.pyplot as plt

# Importar o dataset
df = pd.read_excel(r"C:\Users\tabat\Documents\GitHub\estatisticas\Tempo de uso de redes sociais.xlsx")

# Exibir as primeiras linhas do DataFrame
print(df.head())

# Calcular tempo médio de uso de redes sociais
media = df['Tempo em redes sociais'].mean()
print(f"Tempo médio de uso de redes sociais: {media}")

# Calcular a mediana de uso de redes sociciais
mediana = df['Tempo em redes sociais'].median()
print(f"Mediana de uso de redes sociais: {mediana}")

# Calcular a moda de uso de redes sociais
moda = df['Tempo em redes sociais'].mode()
print(f"Moda de uso de redes sociais: {moda}")

# Calcular a amplitude
amplitude = df['Tempo em redes sociais'].max() - df['Tempo em redes sociais'].min()
print(f"Amplitude: {amplitude}")

# Calcular o número de classes (regra da raiz quadrada, arredondado)
n = df['Tempo em redes sociais'].count()
k = int(round(n ** 0.5))  # Arredondar
print(f"Número de classes: {k}")

# Calcular a amplitude de cada classe
amplitude_classe = math.ceil(amplitude / k)
print(f"Amplitude de cada classe: {amplitude_classe}")

# Criar os limites das classes
min_tempo = df['Tempo em redes sociais'].min()
max_tempo = df['Tempo em redes sociais'].max()
limites_classes = [min_tempo]

while limites_classes[-1] < max_tempo:
    limites_classes.append(limites_classes[-1] + amplitude_classe)

# Ajustar o último limite para cobrir o valor máximo, se necessário
if limites_classes[-1] < max_tempo:
    limites_classes[-1] = max_tempo

print(f"Limites das classes: {limites_classes}")

# Criar as classes e calcular a frequência
df['Classe'] = pd.cut(df['Tempo em redes sociais'], bins=limites_classes, right=False)
frequencia = df['Classe'].value_counts().sort_index()

# Criar o DataFrame com as classes e frequências
classes_df = pd.DataFrame({
    'Limite Inferior': [interval.left for interval in frequencia.index],
    'Limite Superior': [interval.right for interval in frequencia.index],
    'Frequência': frequencia.values
})

# Calcular os pontos médios dos intervalos
classes_df['Ponto Médio'] = (classes_df['Limite Inferior'] + classes_df['Limite Superior']) / 2

# Calcular a média ponderada
numerador = (classes_df['Ponto Médio'] * classes_df['Frequência']).sum()
denominador = classes_df['Frequência'].sum()
media_ponderada = numerador / denominador

print(f"Média Ponderada: {media_ponderada}")

# Construir o histograma
plt.figure(figsize=(10, 6))
plt.bar(
    x=[f"[{row['Limite Inferior']}, {row['Limite Superior']})" for _, row in classes_df.iterrows()],
    height=classes_df['Frequência'],
    color='skyblue',
    edgecolor='black',
    alpha=0.7
)
plt.xlabel('Intervalos de Tempo (em redes sociais)')
plt.ylabel('Frequência')
plt.title('Histograma de Tempo de Uso de Redes Sociais')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()