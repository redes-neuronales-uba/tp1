import neural as neural
import numpy as np
import random


# Datos de entrenamiento para AND, OR y XOR
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


def test_neural_network_back_propagation():
    a_neural_1 = neural.NeuralLayer(3, np.array([[0.8, 0.4, 0.3], [0.2, 0.9, 0.5]]))
    a_neural_2 = neural.NeuralLayer(1, np.array([[0.3], [0.5], [0.9]]))
    n = neural.NeuralNetwork()
    n.add_layer(a_neural_1)
    n.add_layer(a_neural_2)

    for i in range(1, 100):
        x, y_expected = random.choice(training_data_xor)
        n.forward_propagation(np.array([x]))
        n.back_propagation(np.array([y_expected]))

    print(n.forward_propagation(np.array([[1, 0]])))
    print(n.forward_propagation(np.array([[0, 1]])))
    print(n.forward_propagation(np.array([[0, 0]])))
    print(n.forward_propagation(np.array([[1, 1]])))

    return

#TESTS
test_neural_network_back_propagation()
