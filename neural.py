import matplotlib.pyplot as plt
import numpy as np


# Sigmoidea
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoide primera derivada
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Clase para una capa neuroal
class NeuralLayer:
    # Cantidad de neuronas de la capa
    neurons = 0

    # Pesos que se conectan a cada neurona de la capa
    weights = None

    # Induced local field de las neuronas (pesos x input)
    v = None

    # Bias
    bias = None

    #Error total
    total_error = 0

    # Iniciador de neurona (digase constructor)
    def __init__(self, neurons, weights_in=[], bias=(0, 0)):
        #n, m = np.shape(weights_in)
        #assert (n == neurons) # el problema es que numpy no dif entre vectores columna y fila
        self.weights = weights_in        
        self.bias = bias

        # Seteo el numero de neuronas
        self.neurons = neurons

    # Calcula el induced local field, v y lo guarda
    def compute_v(self, x_input):
        self.v = np.dot(self.weights, x_input) + (self.bias[0] * self.bias[1])

    # Devuelve el induced local field v
    def get_v(self):
        return self.v

    # Devuelve el output de las neuronas de esta capa
    def get_output(self, phi):
        return phi(self.v)

    # Devuelve los pesos conectando las neuronas de esta capa con la anterior
    def get_weights(self):
        return self.weights

    # Setea el arreglo de pesos
    def set_weights(self, new_weights):
        self.weights = new_weights
    
    def set_bias(self, bias):
        self.bias = bias

    def get_bias(self, bias):
        return self.bias


class NeuralNetwork:
    # Variables de instancia
    eta = None
    layers = None
    layers_number = None
    X = None
    phi = None

    # Constructor
    def __init__(self, in_signals, eta=0.5, phi=sigmoid):

        # Defino el coeficiente de aprendizaje
        self.eta = eta

        # Lista de valores de errores
        self.errors = []

        # Lista de Capas
        self.layers = []

        # Numero de capas sin contar la de input
        self.layers_number = 0

        # Funcion de activacion
        self.phi = phi

        # Cantidad de seniales de entrada
        self.in_signals = in_signals

    
    
    # Agrega una capa neuronal
    def add_layer(self, layer):
        self.layers.append(layer)
        self.layers_number += 1

    def connect(self): # quizas es mejor hacerlo directamente en el init?
        assert(self.layers > 0)
        factor = 4.0 # para que los pesos esten entre 0 y 1/factor
        neuronas_capa_anterior = self.in_signals
        for i in range(0, self.layers_number):
            capa_a_conectar = self.layers[i]
            filas = capa_a_conectar.neurons
            columnas = neuronas_capa_anterior
            weights = np.random.rand(filas, columnas)/factor
            capa_a_conectar.set_weights(weights)
            capa_a_conectar.set_bias((1, np.random.random()/factor))
            neuronas_capa_anterior = capa_a_conectar.neurons


    # Proceso de forward_propagation para la entrada X
    def forward_propagation(self, X):
        # Tiene que tener al menos 1 capa ademas de la salida
        self.X = X # sirve para algo?
        in_parameters = X
        assert (self.layers_number >= 2) # creo que deberia ser 1
        for i in range(0, len(self.layers)):
            self.layers[i].compute_v(in_parameters)
            in_parameters = self.layers[i].get_output(self.phi)
        return in_parameters

    def get_total_error(self, y_expected):
        output_layer = self.layers[self.layers_number-1]
        errors = (1/2) * ((output_layer.get_output(self.phi) - y_expected)**2)
        return np.sum(errors)


    # Back Propagation para la entrada self.X (definida en forward_propagation) para el resultado esperado y_expected
    def back_propagation(self, y_expected):
        #Solo funciona para 1 capa oculta
        if self.layers_number == 2:
            # Calculo djdW1
            y_estimated = sigmoid(self.layers[1].get_result_of_nodes_sumatory())
            diff_y_expected_and_estimated = -1 * np.subtract(y_expected, y_estimated)
            z3 = self.layers[1].get_result_of_nodes_sumatory()
            f_prime_on_z3 = sigmoid_prime(z3)
            delta_3 = np.multiply(diff_y_expected_and_estimated, f_prime_on_z3)
            a_2 = sigmoid(self.layers[0].get_result_of_nodes_sumatory())
            djdW2 = a_2.T * delta_3
            
            # Calculo djdW2
            W2 = self.layers[1].get_array_weights_in()
            z2 = self.layers[0].get_result_of_nodes_sumatory()
            f_prime_on_z2 = sigmoid_prime(z2)
            delta_2 = np.dot(delta_3, W2.T) * f_prime_on_z2
            djdW1 = np.dot(self.X.T, delta_2)
            
            # Actualizo los pesos de cada capa
            a_array_of_new_weights_1 = self.layers[0].get_array_weights_in() - self.eta * djdW1
            a_array_of_new_weights_2 = self.layers[1].get_array_weights_in() - self.eta * djdW2
            self.layers[0].set_array_weights_in(a_array_of_new_weights_1)
            self.layers[1].set_array_weights_in(a_array_of_new_weights_2)
            return a_array_of_new_weights_1, a_array_of_new_weights_2
