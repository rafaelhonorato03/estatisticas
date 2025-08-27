import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

lstx2 = (100, 125, 125, 150, 150, 200, 200, 250, 250, 300, 300, 350, 400, 400)
lsty2 = (150, 140, 180, 210, 190, 320, 280, 400, 430, 440, 390, 600, 610, 670 )

# Construir o DataFrame e nomear as colunas
df = pd.DataFrame(list(zip(lstx2, lsty2)), columns = ["x","y"])

x = df['x']
y = df['y']

corr = df.corr().iloc[0,1]

print(f'Correlação entre x e y: {corr}')
sns.scatterplot(x=x, y=y)
plt.title('Gráfico de Dispersão entre x e y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(df.info())
print(df.describe())
print(df.head())
print(df.corr())

