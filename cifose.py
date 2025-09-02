import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Idade em meses
x = [12, 15, 42, 52, 59, 73, 82, 91, 96, 105, 114, 120, 121, 128, 130,
     139, 139, 157, 1, 1, 2, 8, 11, 18, 22, 31, 37, 61, 72, 81, 97, 112,
     118, 127, 131, 140, 151, 159, 177, 206]

# Cifose (1 = tem, 0 = não tem)
y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Criar DataFrame
df = pd.DataFrame({'idade_meses': x, 'cifose': y})

# Scatterplot
sns.scatterplot(data=df, x='idade_meses', y='cifose')
plt.title('Idade vs Cifose')
plt.show()

# Modelo Logit
X = sm.add_constant(df['idade_meses'])  # adiciona constante para o intercepto
y = df['cifose']

modelo = sm.Logit(y, X).fit()
print(modelo.summary())
