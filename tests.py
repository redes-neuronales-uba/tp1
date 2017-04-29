import neural as neural
import numpy as np

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


def test_neural_layer():
    a_neural = neural.NeuralLayer(3,np.array([[0.8, 0.2], [0.4, 0.9], [0.3, 0.5]]))
    suma = a_neural.apply_weights_to_params(np.array([1, 1]))
    signal = a_neural.apply_activation_function_to_nodes(lambda x: (1 / (1 + np.exp(-x))))
    #assert(np.array_equal(signal,[0.713105858, 0.78583498,0.68997448]))


def test_neural_network_forward_propagation():
    #a_neural_1 = neural.NeuralLayer(2, np.array([[1, 0],[0,1]]))
    a_neural_2 = neural.NeuralLayer(3, np.array([[0.8,0.4,0.5], [0.2, 0.9, 0.5]]))
    a_neural_3 = neural.NeuralLayer(1, np.array([[0.3, 0.5, 0.9]]))

    n = neural.NeuralNetwork(lambda x:1 / (1 + np.exp(-x)))
    n.addLayer(a_neural_2)
    n.addLayer(a_neural_3)
    #[ 0.77438027]
    print(n.forward_propagation(np.array([1,1])))

def test_neural_network_back_propagation():
    #a_neural_1 = neural.NeuralLayer(2, np.array([[1, 0],[0,1]]))
    a_neural_2 = neural.NeuralLayer(3, np.array([[0.8,0.4,0.3], [0.2, 0.9, 0.5]]))
    a_neural_3 = neural.NeuralLayer(1, np.array([[0.3],[0.5],[0.9]]))

    n = neural.NeuralNetwork(lambda x:1 / (1 + np.exp(-x)))
    n.addLayer(a_neural_2)
    n.addLayer(a_neural_3)
    n.forward_propagation(np.array([[1,1]]))

    n.back_propagation(np.array([0]))



##TESTS
#test_neural_layer()
#test_neural_network_forward_propagation()
test_neural_network_back_propagation()
