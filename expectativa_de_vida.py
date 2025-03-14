import pandas as pd

# Importar o dataset
df = pd.read_csv(r'C:\Users\tabat\Documents\GitHub\estatisticas\life_expectancy.csv')

print(df.head())
print(df.info())

# Remover as linhas onde a expectativa de vida é maior que 100
df = df[df['Sum of Life Expectancy  (both sexes)'] <= 100]
df = df[df['Sum of Females  Life Expectancy'] <= 100]
df = df[df['Sum of Males  Life Expectancy'] <= 100]


# Calcular a amplitude geral da expectativa de vida
amplitude_vida_feminina = df['Sum of Females  Life Expectancy'].max() - df['Sum of Females  Life Expectancy'].min()
amplitude_vida_masculina = df['Sum of Males  Life Expectancy'].max() - df['Sum of Males  Life Expectancy'].min()
amplitude_vida_geral = df['Sum of Life Expectancy  (both sexes)'].max() - df['Sum of Life Expectancy  (both sexes)'].min()

print(df['Sum of Life Expectancy  (both sexes)'].max())
print(df['Sum of Life Expectancy  (both sexes)'].min())


print(f'Amplitude da expectativa de vida feminina: {amplitude_vida_feminina}')
print(f'Amplitude da expectativa de vida masculina: {amplitude_vida_masculina}')   
print(f'Amplitude da expectativa de vida geral: {amplitude_vida_geral}')
