#!/usr/bin/python

import random
from mlp import *

def create_haykin_xor():
    print 2
    rn = MLP([2,2,1])
    
    capa1 = rn.layers[1]
    capa1.neurons[0].weights = [1,1]
    capa1.neurons[0].bias = -1.5
    capa1.neurons[1].weights = [1,1]
    capa1.neurons[1].bias = -0.5
    
    capa2 = rn.layers[2]
    capa2.neurons[0].weights = [-2, 1]
    capa2.neurons[0].bias = -0.5
    
    rn.print_MLP()

    print "Resultados"
    print "[1,1] -> esperado" , 0, "calculado ", rn.forward([1,1])
    print "[1,0] -> esperado" , 1, "calculado ", rn.forward([1,-1])
    print "[0,1] -> esperado" , 1, "calculado ", rn.forward([-1,1])
    print "[0,0] -> esperado" , 0, "calculado ", rn.forward([-1,-1])
    
lset_xor = [
        ([1,1], [0.0001]),
        ([1,0], [0.99]),
        ([0,1], [0.99]),
        ([0,0], [0.0001])
        ]
    
lset_xor_and_or = [
                    ([1,1], [0.0001, 0.99, 0.99]),
                    ([1,0], [0.99, 0.001, 0.99]),
                    ([0,1], [0.99, 0.001, 0.99]),
                    ([0,0], [0.0001, 0.001, 0.001])
                ]

def train_xor_and_or(it, eta, a, desvio):
    # defino arquitectura
    rn = MLP([2,2,3], phi=logistica(a), phi_prima=logistica_p(a), desvio=desvio)
    
    # entreno
    for i in range(it):
        x, d = random.choice(lset_xor_and_or)
        rn.backprop(x, d, eta)
        
    print "[1,1] -> esperado" , 0, 1, 1, "calculado ", rn.forward([1,1])
    print "[1,0] -> esperado" , 1, 0, 1, "calculado ", rn.forward([1,0])
    print "[0,1] -> esperado" , 1, 0, 1, "calculado ", rn.forward([0,1])
    print "[0,0] -> esperado" , 0, 0, 0, "calculado ", rn.forward([0,0])


train_xor_and_or(10000, 0.7, 10, 1)    # entrena, parece que lo que estaba mal era el training set. 
                                # lo cambie para que la salida sea 0.0001 y 0.99 !!!!!
                                # tambien cambie la logistica para que sea mas parecida a la heaveside
                                # requiere 10000 iteraciones, que e parece demasiado...

