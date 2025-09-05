# =========================================================
# Regressão Polinomial com Statsmodels
# Versão Melhorada, Corrigida e Generalizada
# =========================================================

# 1. Importação das bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# 2. Configuração do estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')

# 3. Dados experimentais
velocidade = [7, 10.3, 13.7, 16.6, 19.8, 22]
goticulas = [479, 503, 487, 470, 458, 412]
df = pd.DataFrame({'Velocidade': velocidade, 'Gotículas': goticulas})

# =========================================================
# Funções auxiliares
# =========================================================

def ajustar_modelo(df, grau=2):
    """Ajusta um modelo polinomial de grau definido."""
    poly = PolynomialFeatures(degree=grau, include_bias=False)
    X_poly = poly.fit_transform(df['Velocidade'].values.reshape(-1, 1))
    X_poly = sm.add_constant(X_poly)  # adiciona intercepto
    modelo = sm.OLS(df['Gotículas'], X_poly).fit()
    return modelo, poly

def prever(modelo, poly, valor, alpha=0.05):
    """Retorna previsão e intervalo de confiança para um valor específico."""
    # Gera features polinomiais
    X_val = poly.transform(np.array([[valor]]))
    # Adiciona constante para alinhar com o modelo
    X_val = sm.add_constant(X_val, has_constant='add')
    # Faz previsão com intervalo de confiança
    pred = modelo.get_prediction(X_val).summary_frame(alpha=alpha)
    return pred

def plotar_modelo(df, modelo, poly):
    """Plota dados, curva ajustada e resíduos."""
    # Curva ajustada
    velocidade_pred = np.linspace(min(df['Velocidade']), max(df['Velocidade']), 200)
    X_pred = sm.add_constant(poly.transform(velocidade_pred.reshape(-1, 1)))
    y_pred = modelo.predict(X_pred)

    # Gráfico principal
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Velocidade'], df['Gotículas'], color='black', s=80, label='Dados Observados')
    plt.plot(velocidade_pred, y_pred, color='green', linewidth=3,
             label=f'Modelo Polinomial (R²: {modelo.rsquared:.3f})')
    plt.title('Relação entre Velocidade do Fluxo e Gotículas de Névoa')
    plt.xlabel('Velocidade do Fluxo (cm/s)')
    plt.ylabel('Quantidade de Gotículas (mg/m³)')
    plt.legend()
    plt.annotate(
        'Curvatura ajusta bem os dados.',
        xy=(17, 470), xytext=(18, 490),
        arrowprops=dict(facecolor='green', shrink=0.05),
        fontsize=12, color='green'
    )
    plt.show()

    # Gráfico de resíduos
    residuos = modelo.resid
    plt.figure(figsize=(10, 4))
    plt.scatter(df['Velocidade'], residuos, color='blue', s=70)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Resíduos do Modelo')
    plt.xlabel('Velocidade do Fluxo (cm/s)')
    plt.ylabel('Resíduo')
    plt.show()

# =========================================================
# Execução da análise
# =========================================================

# Ajuste do modelo
grau = 2
modelo, poly = ajustar_modelo(df, grau=grau)

# Resumo estatístico
print("\n### Resumo da Regressão Polinomial ###")
print(modelo.summary())

# Visualização
plotar_modelo(df, modelo, poly)

# Previsão para exemplo
velocidade_exemplo = 13.7
pred = prever(modelo, poly, velocidade_exemplo)

print("\n--- Previsão de Exemplo ---")
print(f"Velocidade: {velocidade_exemplo} cm/s")
print(f"  Média Prevista: {pred['mean'][0]:.2f} mg/m³")
print(f"  Intervalo de Confiança 95%: [{pred['mean_ci_lower'][0]:.2f}, {pred['mean_ci_upper'][0]:.2f}] mg/m³")
