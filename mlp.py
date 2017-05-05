#!/usr/bin/python

import numpy as np
from itertools import *

def logistica(a):
    return lambda x: 1 / (1 + np.exp(-a*x))


def logistica_p(a):
    return lambda x: a*logistica(a)(x)*(1 - logistica(a)(x))

def create_random_matrix(filas, columnas, desvio):
    M1 = np.random.rand(filas, columnas)*desvio
    M2 = -1*np.random.rand(filas, columnas)*desvio
    M = M1 + M2
    return np.array(M)

def create_random_bias(size, factor):
    return np.random.rand(size)*factor

class MLP:

    def __init__(self, neurons, eta=0.1, phi=logistica(1), phi_prima=logistica_p(1), desvio_coef=0.1):

        self.phi = phi
        self.phi_prima = phi_prima
        self.eta = eta
        self.depth = len(neurons)-1

        self.weights = []
        self.bias = []
        for i in range(0, self.depth):
            self.weights.append(create_random_matrix(neurons[i+1], neurons[i], desvio_coef))
            self.bias.append(create_random_bias(neurons[i+1], desvio_coef))

        self.v = [] # contiene los campos inducidos en forma vectorial
        self.y = [] # contiene los outputs de las neuronas en forma vectorial
                    # y[0] = input de la red, y[self.depth] = output de la red

        

    def feedForward(self, x):
        assert(len(x) == np.shape(self.weights[0])[1])

        phi = self.phi

        x = np.array(x)
        self.y.append(x)

        for W, b in izip(self.weights, self.bias):
            self.v.append(np.dot(W, self.y[-1]) + b)
            self.y.append(phi(self.v[-1]))
        
        return self.y[-1]

    def backprop(self, x, d):
        assert(len(d) == np.shape(self.weights[-1])[0])
        
        phi_p = self.phi_prima
        d = np.array(d)

        # computo el output y el induced local field de todas las neuronas
        self.feedForward(x) # ahora self.y[-1] tiene el output de la red para x

        # computo el primer delta
        delta = []
        error = d - self.y[-1]
        delta.append(error * phi_p(self.v[-1]))

        # computo los deltas de las capas ocultas
        for i in reversed(range(self.depth-1)):
            W = self.weights[i+1]
            v = self.v[i]
            delta.insert(0, phi_p(v) * np.dot(W.T, delta[0])) #

        # actualizo los bias de todas las capas
        self.bias = [b + self.eta*d for b, d in izip(self.bias, delta)]
        
        # actualizo los pesos de todas las capas
        for i in range(self.depth):
            self.weights[i] = self.weights[i] + self.eta*(delta[i].reshape(-1, 1) * self.y[i])

        #return self.weights

    def set_wyb(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def print_y(self):
        print self.y
    
    # Funciones usadas para debug
    def print_wyb(self):
        print "W's"
        for w in self.weights:
            print w

        print "b's"
        for b in self.bias:
            print b


