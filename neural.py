import matplotlib.pyplot as plt
import numpy as np


class NeuralLayer:
    internal_neurals_number = 0
    array_weights_in = []
    array_nodes_sumatory = None
    array_nodes_apply_activation_function = None

    def __init__(self, internal_neurals_number, weights_in=[]):
        #Si no me dan pesos los pongo aleatorios
        if(len(weights_in)==0):
            self.array_weights_in = np.random.rand(internal_neurals_number)
        else:
            self.array_weights_in = weights_in

        #Seteo el numero de neuronas
        self.internal_neurals_number = internal_neurals_number

    def apply_weights_to_params(self, values_matrix):
        self.array_nodes_sumatory = np.dot(self.array_weights_in, values_matrix)
        return self.array_nodes_sumatory

    def apply_activation_function_to_nodes(self, activation_function):
        assert(not(self.array_nodes_sumatory is None))
        array_nodes_activation_function_result = np.zeros((self.internal_neurals_number))
        for i in range(0, self.internal_neurals_number):
            array_nodes_activation_function_result[i] = activation_function(self.array_nodes_sumatory[i])
        return array_nodes_activation_function_result



class NeuralNetwork:

    # Funciones de activacion
    FUNC_UNIT_STEP = lambda x: 0 if x < 0 else 1
    def sigmoid(self,x):
        return (1 / (1 + np.exp(-x)))

    def sigmoid_deriv(self,x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    # Variables de instancia
    training_set = None
    epochs = None
    eta = None
    activation_function = None
    errors = None
    weights = None
    hidden_layer_size = None
    hidden_layer = None
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

    # Entrena la red
    def train(self, training_set, epochs=100):

        #Defino el trainning set
        self.training_set = training_set

        #Defino la cantidad de epocas
        self.epochs = epochs

        #Inicializando pesos
        parameters_size = len(self.training_set[0][0])
        self.weights = np.array([[0.8, 0.2], [0.4, 0.9], [0.3, 0.5]])
        #self.weights = np.random.rand(self.hidden_layer_size, self.hidden_layer_size)
        self.hidden_layer_weights = np.array([0.3, 0.5, 0.9])
        #self.hidden_layer_weights = np.random.rand(self.hidden_layer_size)
        self.hidden_layer = np.zeros((1, self.hidden_layer_size))

        #Entrenar
        #for i in range(self.epochs):
            #x, expected = random.choice(self.training_set)
            #result = np.dot(self.weights, x)
            #error = expected - self.activation_function(result)
            #self.errors.append(error)
            #self.weights += self.eta * error * x

    # Evalua un arreglo de valores e imprime los resultados
    def evaluate(self, array_values=[]):
        for x, _ in array_values:
            result = np.dot(x,self.weights)
            print("{}: {} -> {}".format(x[:2], result, self.activation_function(result)))

    # Grafica el error con respecto a las epocas
    def draw_error(self):
        plt.plot(self.errors)
        plt.show()

    def forward_propagation(self, in_parameters):
        #Tiene que tener al menos 2 capas, la entrada y la salida
        assert(len(self.layers)>=2)
        for i in range(0, len(self.layers)):
            layer_sums = self.layers[i].apply_weights_to_params(in_parameters)
            in_parameters = self.layers[i].apply_activation_function_to_nodes(self.activation_function)
        return in_parameters

    def back_propagation(self):
        x, expected = self.training_set[3]
        delta_output_sum = self.sigmoid_deriv(self.sum_result) * (expected - self.forward_propagation_bkp())
        delta_weights = np.dot(delta_output_sum, self.hidden_layer_result)
        delta_hidden_sum = self.hidden_layer
        for i in range(0, len(self.hidden_layer)):
            delta_hidden_sum[i] = self.sigmoid_deriv(self.hidden_layer[i])
        hidden_layer_delta_weights = delta_output_sum * self.hidden_layer_weights * delta_hidden_sum
#        self.weights = np.subtract(delta_weights, delta_weights * x)
#        self.hidden_layer_weights = np.subtract(self.hidden_layer_weights, hidden_layer_delta_weights)
#        return 0

    def addLayer(self, a_neural_layer):
        self.layers.append(a_neural_layer)




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
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 0),
]