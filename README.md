# Estatísticas

Este projeto contém scripts e análises estatísticas utilizando Python e bibliotecas como Pandas, NumPy e Matplotlib. Ele foi desenvolvido para realizar cálculos estatísticos, gerar gráficos e exportar os resultados para arquivos Excel.

## Descrição

O objetivo deste projeto é realizar análises estatísticas em dados gerados aleatoriamente ou provenientes de datasets públicos. As análises incluem:
- Contagem de frequências.
- Cálculo de frequências relativas e acumuladas.
- Geração de gráficos como histogramas, polígonos de frequência e ogivas.
- Exportação dos resultados para arquivos Excel.

## Estrutura do Projeto

- **`projecao_idades.py`**: Script que gera números aleatórios, calcula frequências e salva os resultados em um arquivo Excel.
- **`estatisticas_idades.py`**: Script que realiza análises estatísticas em um dataset de idades, incluindo a criação de classes, cálculo de frequências e geração de gráficos (histograma, polígono de frequência e ogiva).
- **`expectativa_de_vida.py`**: Script que analisa dados de expectativa de vida, removendo outliers e calculando amplitudes.
- **`tempo_de_uso_redes_sociais.py`**: Script que analisa o tempo de uso de redes sociais, criando classes, calculando frequências e gerando gráficos como histogramas.
- **`frequencia.xlsx`**: Arquivo Excel gerado contendo as frequências calculadas.
- **`frequencia_agrupada.xlsx`**: Arquivo Excel gerado contendo as frequências agrupadas por intervalos de classes.

## Requisitos

Para executar os scripts, você precisará dos seguintes requisitos:

- **Python 3.x**
- **Bibliotecas Python**:
  - Pandas
  - NumPy
  - Matplotlib
  - Openpyxl (para salvar arquivos Excel)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/estatisticas.git
   cd estatisticas
   pip install pandas numpy matplotlib openpyxl
   ```

Data_set retirado do Kaggle: https://www.kaggle.com/datasets/ignacioazua/life-expectancy?resource=download
