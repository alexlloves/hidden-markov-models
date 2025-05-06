from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.markov_chain import MarkovChain
import torch

'''
Ahora, las habitaciones 2 y 3, y 4 y 5 están conectadas.
Sin embargo, a la habitación 6 sólo puede accederse
desde la habitación 3.
Distribuímos las probabilidades de manera igual al inicio.
La probabilidad de permanecer en la misma habitación es 1/5,
y la de cambiar de habitación 2/5.
'''

# Probabilidades iniciales
start = Categorical([[
    1/6,
    1/6,
    1/6,
    1/6,
    1/6,
    1/6
]])

# Modelo de transicion
transitions = ConditionalCategorical([[
    [1/5, 2/5, 0, 2/5, 0, 0], #probabilidad de pasar de la habitacion 1 a la 1, de la 1 a la 2, de la 1 a la 3, de la 1 a la 4...
    [2/5, 1/5, 2/5, 0, 0, 0], #probabilidad de pasar de la habitacion 2 a la 1, de la 2 a la 2...
    [0, 2/5, 1/5, 0, 0, 2/5],
    [2/5, 0, 0, 1/5, 2/5, 0],
    [0, 0, 0, 4/5, 1/5, 0],
    [0, 0, 4/5, 0, 0, 1/5]
]], [start])

# Crear Markov chain
model = MarkovChain([start, transitions], k=3)

print("Probabilidades iniciales:")
# Probabilidades iniciales (Categorical)
print(model.distributions[0].probs[0])
# Probabilidades de transición o condicionadas (ConditionalCategorical)
print("Matriz de transicion:")
print(model.distributions[1].probs[0])

# vector de probabilidades iniciales
v = model.distributions[0].probs[0]

# matriz de transicion
P = model.distributions[1].probs[0]

# Probabilidades en el segundo dia de la serie
w = torch.matmul(v, P)
print(f'w = {w}')

# Probabilidades en el tercer dia de la serie
u = torch.matmul(w, P)
print(f'u = {u}')

# Probabilidades en el cuarto dia de la serie
t = torch.matmul(u, P)
print(f't = {t}')

'''
w = tensor([0.1667, 0.1667, 0.2333, 0.2333, 0.1000, 0.1000])
u = tensor([0.1933, 0.1933, 0.1933, 0.1933, 0.1133, 0.1133])
t = tensor([0.1933, 0.1933, 0.2067, 0.2067, 0.1000, 0.1000])
'''

samples = model.sample(20)
population = [ str(sample[0][0].item() + 1) + " -> " + str(sample[1][0].item() + 1) for sample in samples ]
print(f'Numero de muestras: {len(population)}')
print(" | ".join(population))