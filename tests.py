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


def test_neural_network_back_propagation_xor():
    b1 = (1, 0)
    b2 = (1, 0)
    a_neural_1 = neural.NeuralLayer(3, np.array([[0.1, 0.2], [0.2, 0.01], [0.2, 0.02]]), b1)
    a_neural_2 = neural.NeuralLayer(1, np.array([[0.1, 0.05, 0.1]]), b2)
    n = neural.NeuralNetwork()
    n.add_layer(a_neural_1)
    n.add_layer(a_neural_2)

    for i in range(1, 2):
        x, y_expected = random.choice(training_data_xor)
        n.forward_propagation(np.array(x))
        n.back_propagation(np.array([y_expected]))

    print n.forward_propagation(np.array([1, 0]))
    print n.forward_propagation(np.array([0, 1]))
    print n.forward_propagation(np.array([0, 0]))
    print n.forward_propagation(np.array([1, 1]))

    return

def test_neural_back_propagation_example():
    w_layer_1 = np.array([[0.15, 0.20], [0.25, 0.3]]).T
    w_layer_2 = np.array([[0.4, 0.50], [0.45, 0.55]]).T
    bias_1 = (1, 0.35)
    bias_2 = (1, 0.6)
    a_neural_1 = neural.NeuralLayer(2, w_layer_1, bias_1)
    a_neural_2 = neural.NeuralLayer(2, w_layer_2, bias_2)
    n = neural.NeuralNetwork()
    n.add_layer(a_neural_1)
    n.add_layer(a_neural_2)

    x = np.array([[0.05, 0.10]])
    y_expected = np.array([[0.01, 0.99]])
    for i in range(1, 10000):
        print n.forward_propagation(x)
        print n.get_total_error(y_expected)
        n.back_propagation(y_expected)


def test_forward_mazzur():
    # Test del ejemplo en 
    # mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    l1 = np.array([[0.15, 0.20], [0.25, 0.30]])
    l2 = np.array([[0.4, 0.45], [0.50, 0.55]])
    b1 = (1, 0.35)
    b2 = (1, 0.6)
    nl1 = neural.NeuralLayer(2, l1, b1)
    nl2 = neural.NeuralLayer(2, l2, b2)

    n = neural.NeuralNetwork(2)
    n.add_layer(nl1)
    n.add_layer(nl2)

    # El valor correcto es [0.75136507, 0.772928465]
    print n.forward_propagation(np.array([0.05, 0.10]))


def test_connect():
    # Testeo que las dimensiones de todas las matrices estan bien.
    nl1 = neural.NeuralLayer(2)
    nl2 = neural.NeuralLayer(2)

    n = neural.NeuralNetwork(2)
    n.add_layer(nl1)
    n.add_layer(nl2)
    n.connect()

    print n.forward_propagation(np.array([0.05, 0.10]))


def test_forward_miller():
        # Test del ejemplo en 
        # stevenmiller888.github.io/mind-how-to-build-a-neural-network/
        l1 = np.array([[0.8, 0.2], [0.4, 0.9], [0.3, 0.5]])
        l2 = np.array([[0.3, 0.5, 0.9]])
        b1 = (0, 0)
        b2 = (0, 0)
        nl1 = neural.NeuralLayer(3, l1, b1)
        nl2 = neural.NeuralLayer(1, l2, b2)
        
        n = neural.NeuralNetwork(2)
        n.add_layer(nl1)
        n.add_layer(nl2)

        x = np.array([1,1])
        # El valor correcto es 0.77 (el autor no da mas decimales)
        print n.forward_propagation(x)

# Test connect
test_connect()

# Tests forward
#test_forward_mazzur()
#test_forward_miller()

# Test backward
#test_neural_network_back_propagation()
#test_neural_back_propagation_example()
#test_neural_network_back_propagation_xor()

