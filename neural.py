import matplotlib.pyplot as plt
import numpy as np
import random


class NeuralNetwork:

    # Funciones de activacion
    FUNC_UNIT_STEP = lambda x: 0 if x < 0 else 1
    FUNC_SIGMOID = lambda x: 1 / (1 + np.exp(-x))

    # Variables de instancia
    training_set = None
    epochs = None
    eta = None
    activation_function = None
    errors = None
    weights = None

    # Constructor
    def __init__(self, eta=0.2, activation_function=FUNC_UNIT_STEP):

        #Defino el coheficiente de aprendizaje
        self.eta = eta

        #Defino la funcion de activacion
        self.activation_function = activation_function

        #Lista de valores de errores
        self.errors = []

    # Entrena la red
    def train(self, training_set, epochs=100):

        #Defino el trainning set
        self.training_set = training_set

        #Defino la cantidad de epocas
        self.epochs = epochs

        #Valor inicial
        parameters_size = len(self.training_set[0][0])
        self.weights = np.random.rand(parameters_size)

        #Entrenar
        for i in range(self.epochs):
            x, expected = random.choice(self.training_set)
            result = np.dot(self.weights, x)
            error = expected - self.activation_function(result)
            self.errors.append(error)
            self.weights += self.eta * error * x

    # Evalua un arreglo de valores e imprime los resultados
    def evaluate(self, array_values=[]):
        for x, _ in array_values:
            result = np.dot(x,self.weights)
            print("{}: {} -> {}".format(x[:2], result, self.activation_function(result)))

    # Grafica el error con respecto a las epocas
    def draw_error(self):
        plt.plot(self.errors)
        plt.show()


######################

# Datos de entrenamiento
training_data_or = [
    (np.array([0, 0, 1]), 0),
    (np.array([0, 1, 1]), 1),
    (np.array([1, 0, 1]), 1),
    (np.array([1, 1, 1]), 1),
]

training_data_and = [
    (np.array([0, 0, 1]), 0),
    (np.array([0, 1, 1]), 0),
    (np.array([1, 0, 1]), 0),
    (np.array([1, 1, 1]), 1),
]

training_data_xor = [
    (np.array([0, 0, 1]), 0),
    (np.array([0, 1, 1]), 1),
    (np.array([1, 0, 1]), 1),
    (np.array([1, 1, 1]), 0),
]


n = NeuralNetwork()
n.train(training_data_and, 150)
n.evaluate(training_data_or)
n.draw_error()

