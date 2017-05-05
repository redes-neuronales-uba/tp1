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
    def get_output(self, phi=sigmoid):
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
        factor = 6.0 # para que los pesos esten entre 0 y 1/factor
        neuronas_capa_anterior = self.in_signals
        for i in range(0, self.layers_number):
            capa_a_conectar = self.layers[i]
            filas = capa_a_conectar.neurons
            columnas = neuronas_capa_anterior
            weights = (np.random.rand(filas, columnas) - np.random.rand(filas, columnas))/(2*factor)
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


    def back_propagation(self, y_expected):
        #calculo el delta para la capa del output
        e1 = y_expected - self.layers[1].get_output(self.phi)
        sigma1 = np.multiply(e1, sigmoid_prime(self.layers[1].get_v()))
        out_capa_anterior = self.layers[0].get_output(self.phi)
        W1 = self.layers[1].get_weights() + self.eta * np.multiply(sigma1.reshape(-1,1), out_capa_anterior)
        Ws = [W1]
        
        sigma_calculado = sigma1
        for i in range(0, len(self.layers)-1):
            #calculo el delta para la primer capa
            #print sigma_calculado
            #print self.layers[i+1].get_weights()
            p_interno = np.dot(sigma_calculado, self.layers[i+1].get_weights())
            #print p_interno
            sigma_actual = np.multiply(sigmoid_prime(self.layers[i].get_v()), p_interno)
            sigma_calculado = sigma_actual
            #print sigma0
            if i == 0:
                out_capa_anterior = self.X
            else:
                out_capa_anterior = self.layers[i-1].get_output(self.phi)
            #print out_capa_anterior
            Wnew = self.layers[0].get_weights() + self.eta * np.multiply(sigma_actual.reshape(-1,1), out_capa_anterior)
            #print W0
            Ws.insert(0, Wnew)
            
                        
        #Seteo los nuevos pesos        
        for i in range(0, len(self.layers)):
            self.layers[i].set_weights(Ws[i])
        
