import random
from typing import TypeVar, List, Tuple

x = TypeVar('x') # Genérico para representar dados
y = TypeVar('y')

def split_data(data: List[x], prob: float) -> Tuple[List[x], List[x]]:
    '''
    Divida os dados em frações [prob, 1 - prob]
    '''
    data = data[:] #copia superficial
    random.shuffle(data) #shuffle modifica a lista.
    cut = int(len(data) * prob) #usando prob para encontrar um limiar
    return data[:cut], data[cut:] # dividir a lista aleatória nesse ponto

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

assert len(train) == 750
assert len(test) == 250
assert sorted(train + test) == data

def train_test_split(xs: List[x],
                     ys: List[y],
                     test_pct: float) -> Tuple[List[x], List[x], List[y], List[y]]:
    '''
    Divide os dados em treino e teste.
    
    Args:
        xs: Lista de variáveis de entrada (features)
        ys: Lista de variáveis de saída (labels)
        test_pct: Proporção do conjunto de teste (ex: 0.2 -> 20%)
        
    Returns:
        x_train, x_test, y_train, y_test
    '''

    idx = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idx, 1 - test_pct)

    return (
        [xs[i] for i in train_idxs],
        [xs[i] for i in test_idxs],
        [ys[i] for i in train_idxs],
        [ys[i] for i in test_idxs]
    )

data_x = [i for i in range(10)]
data_y = [i**2 for i in range(10)]

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, 0.2)

