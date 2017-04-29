import matplotlib.pyplot as plt
import numpy as np

#Sigmoidea
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

#Sigmoide primera derivada
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

#Clase para una capa neuroal
class NeuralLayer:

    #Cantidad de neuronas de la capa
    internal_neurals_number = 0

    #Pesos que se conectan a cada neurona de la capa
    array_weights_in = None

    #Resultado para cada neurona de aplicar los pesos a los parametros que le van a pasar a la neurona
    array_nodes_sumatory = None

    #Resultado para cada neura de aplicar la funcion de activación a la sumatoria anterior (array_nodes_sumatory)
    array_nodes_apply_activation_function = None

    #Iniciador de neurona (digase constructor)
    def __init__(self, internal_neurals_number, weights_in=[]):
        #Si no me dan pesos los pongo aleatorios
        if(len(weights_in)==0):
            self.array_weights_in = np.random.rand(internal_neurals_number)
        else:
            self.array_weights_in = weights_in

        #Seteo el numero de neuronas
        self.internal_neurals_number = internal_neurals_number

    def apply_weights_to_params(self, values_matrix):
        self.array_nodes_sumatory = np.dot(values_matrix, self.array_weights_in)
        return self.array_nodes_sumatory

    def apply_activation_function_to_nodes(self, activation_function):
        assert(not(self.array_nodes_sumatory is None))
        n,m = np.shape(self.array_nodes_sumatory)
        self.array_nodes_apply_activation_function = np.zeros((n,m))
        for i in range(0, n):
            vfunc = np.vectorize(activation_function)
            self.array_nodes_apply_activation_function[i] = vfunc(self.array_nodes_sumatory[i])
        return self.array_nodes_apply_activation_function

    def get_result_of_apply_activation_function(self):
        return self.array_nodes_apply_activation_function

    def get_result_of_nodes_sumatory(self):
        return self.array_nodes_sumatory

    def get_array_weights_in(self):
        return self.array_weights_in

class NeuralNetwork:
    # Variables de instancia
    training_set = None
    epochs = None
    eta = None
    activation_function = None
    errors = None
    weights = None
    layers = None

    # Constructor
    def __init__(self,activation_function, hidden_layer_size=3, eta=0.2 ):

        #Defino el coheficiente de aprendizaje
        self.eta = eta

        #Defino la funcion de activacion
        self.activation_function = activation_function

        #Lista de valores de errores
        self.errors = []

        #Tamaño de la capa oculta
        self.hidden_layer_size = hidden_layer_size

        #Lista de Capas
        self.layers = []

    # Grafica el error con respecto a las epocas
    def draw_error(self):
        plt.plot(self.errors)
        plt.show()

    def forward_propagation(self, X):
        #Tiene que tener al menos 2 capas, la entrada y la salida
        self.X = X
        in_parameters = X
        assert(len(self.layers)>=2)
        for i in range(0, len(self.layers)):
            self.layers[i].apply_weights_to_params(in_parameters)
            in_parameters = self.layers[i].apply_activation_function_to_nodes(self.activation_function)
        return in_parameters

    def back_propagation(self,  y_expected):
        #Seteo el indice, va de la ultima a la primera
        idx = len(self.layers) - 1

        #Arreglo de deltas
        array_delta = np.zeros(len(self.layers))
        djdW = np.zeros(len(self.layers))

        #Ultima capa
        y_estimated = self.layers[idx].get_result_of_apply_activation_function()
        diff_expected_and_estimated = np.subtract(y_expected, y_estimated)
        vfunction = np.vectorize(sigmoid_prime)
        f_prime_on_nodes_sum = vfunction(self.layers[idx].get_result_of_nodes_sumatory())
        array_delta = np.multiply(diff_expected_and_estimated, f_prime_on_nodes_sum)
        djdW[idx] = np.dot(self.layers[idx-1].get_result_of_apply_activation_function().T, array_delta)

        #Ultima capa - 1
        idx-=1
        array_delta[idx] = np.dot(array_delta[idx+1], self.layers[idx].get_array_weights_in.T) * sigmoid_prime(self.layers[idx].get_result_of_apply_activation_function())
        djdW[idx] = np.dot(self.X.T, array_delta[idx])

        return djdW[1], djdW[0]

    def addLayer(self, a_neural_layer):
        self.layers.append(a_neural_layer)