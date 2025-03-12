import pandas as pd
import numpy as np

numeros_aleatorios = np.random.randint(0, 100, 1000)
print(numeros_aleatorios)

valores, contagem = np.unique(numeros_aleatorios, return_counts=True)
frequencia = pd.DataFrame({'Valores': valores, 'Contagem': contagem})
frequencia['Frequencia Relativa'] = frequencia['Contagem'] / frequencia['Contagem'].sum()
frequencia['Frequencia Acumulada'] = frequencia['Frequencia Relativa'].cumsum()
frequencia['Frequencia Relativa (%)'] = (frequencia['Contagem'] / len(numeros_aleatorios)) * 100
frequencia['Frequencia Acumulada (%)'] = frequencia['Frequencia Relativa (%)'].cumsum()

# Criar intervalos automaticamente usando quantis
frequencia['Intervalo de Classes'] = pd.qcut(frequencia['Valores'], q=5)

print(frequencia)

frequencia.to_excel(r'c:\Users\tabat\Documents\GitHub\estatisticas\frequencia.xlsx', index=False)