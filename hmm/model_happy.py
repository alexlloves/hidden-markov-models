
'''
Tutorial:
https://pomegranate.readthedocs.io/en/latest/tutorials/B_Model_Tutorial_4_Hidden_Markov_Models.html
'''

from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM
import numpy

# Modelo de observaciones para cada estado
sun = Categorical([[
    0.9, # "happy"
    0.1  # "grumpy"
]])

rain = Categorical([[
    0.4,  # "happy"
    0.6   # "grumpy"
]])

# Crear el modelo
model = DenseHMM()
model.add_distributions([sun, rain])

# Probabilidades de inicio
model.add_edge(model.start, sun, 0.5)
model.add_edge(model.start, rain, 0.5)

# Modelo de transiciones
model.add_edge(sun, sun, 0.8) # Prediccion de mañana si hoy = sun
model.add_edge(sun, rain, 0.2)
model.add_edge(rain, sun, 0.4) # Prediccion de mañana si hoy = rain
model.add_edge(rain, rain, 0.6)

# Datos observados / evidencia
observations = [
    "happy",
    "happy",
    "grumpy",
    "happy",
    "happy",
    "happy",
    "happy",
    "grumpy",
    "grumpy"
]

X = numpy.array([[[['happy', 'grumpy'].index(observation)] for observation in observations]])
print(f'Dimensiones del array de observaciones: {X.shape}')
# (1, 9, 1)

# Predecir el estado oculto, el estado del clima
y_hat = model.predict(X)
# tensor([[1, 1, 0, 1, 1, 1, 1, 0, 0]])

hmm_predictions = ["sun" if y.item() == 0 else "rain" for y in y_hat[0]]

for observation, prediction in zip(observations, hmm_predictions):
    print(f'{observation} -> {prediction}')

# print("observaciones:\n {}".format(' '.join(observations)))
# print("hmm pred:\n {}".format(' '.join(["sun" + "->" if y.item() == 0 else "rain" + "->" for y in y_hat[0]])))


# probabilidades posteriores de cada muestra de la serie
proba = model.predict_proba(X)
print(proba[0])

# P(R1 | H1)
print(f"P(R1 | H1) = {proba[0][0][1].item()}")