import numpy as np
np.set_printoptions(threshold=np.inf)
# Criando um array 25x25 preenchido com 1s
window = np.ones((51,51), dtype=int)

# Convertendo o array em uma única linha como uma string
window_line = np.ravel(window)

# Imprimindo a linha como uma string formatada
print(window_line.tolist())