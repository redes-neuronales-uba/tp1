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
    internal_neurals_number = 0

    # Pesos que se conectan a cada neurona de la capa
    array_weights_in = None

    # Resultado para cada neurona de aplicar los pesos a los parametros que le van a pasar a la neurona
    array_nodes_sumatory = None

    # Iniciador de neurona (digase constructor)
    def __init__(self, internal_neurals_number, weights_in=[]):
        # Si no me dan pesos los pongo aleatorios
        if (len(weights_in) == 0):
            self.array_weights_in = np.random.rand(internal_neurals_number)
        else:
            n, m = np.shape(weights_in)
            assert (n != internal_neurals_number)
            self.array_weights_in = weights_in

        # Seteo el numero de neuronas
        self.internal_neurals_number = internal_neurals_number

    # Aplica los pesos a la variable de entrada values_matrix
    def apply_weights_to_params(self, values_matrix):
        self.array_nodes_sumatory = np.dot(values_matrix, self.array_weights_in)

    # Retorna array_nodes_sumatory
    def get_result_of_nodes_sumatory(self):
        return self.array_nodes_sumatory

    # Retorna el arreglo de pesos
    def get_array_weights_in(self):
        return self.array_weights_in

    # Setea el arreglo de pesos
    def set_array_weights_in(self, a_new_array_weights_array):
        self.array_weights_in = a_new_array_weights_array


class NeuralNetwork:
    # Variables de instancia
    eta = None
    layers = None
    layers_number = None
    X = None

    # Constructor
    def __init__(self, eta=0.2):

        # Defino el coheficiente de aprendizaje
        self.eta = eta

        # Lista de valores de errores
        self.errors = []

        # Lista de Capas
        self.layers = []

        # Numero de capas
        self.layers_number = 0

    # Agrega una capa neuronal
    def add_layer(self, a_neural_layer):
        self.layers.append(a_neural_layer)
        self.layers_number += 1

    # Proceso de forward_propagation para la entrada X
    def forward_propagation(self, X):
        # Tiene que tener al menos 2 capas, la entrada y la salida
        self.X = X
        in_parameters = X
        assert (self.layers_number >= 2)
        for i in range(0, len(self.layers)):
            self.layers[i].apply_weights_to_params(in_parameters)
            in_parameters = sigmoid(self.layers[i].get_result_of_nodes_sumatory())
        return in_parameters

    # Back Propagation para la entrada self.X (definida en forward_propagation) para el resultado esperado y_expected
    def back_propagation(self, y_expected):
        #Solo funciona para 1 capa oculta
        if self.layers_number == 2:
            # Calculo djdW1
            y_estimated = sigmoid(self.layers[1].get_result_of_nodes_sumatory())
            diff_y_expected_and_estimated = np.subtract(y_expected, y_estimated)
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
        return