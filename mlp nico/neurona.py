#!/usr/bin/python

import numpy as np

from factivacion import *

class Neuron(object):
    """Implementa una neurona """

    def __init__(self, activation=None, phi=logistica(3), phi_prima=logistica_p(1), weights=[], bias=0):
        self.phi = phi
        self.phi_prima = phi_prima
        self.weights = weights
        self.bias = bias
        self.z = None
        self.a = activation
        self.delta = None
        self.expected = None
        self.position = None


    def compute_z(self, estimulo):
        assert(len(estimulo) == len(self.weights))
        self.z = np.dot(estimulo, self.weights) + self.bias
        return self.z

    def estimular(self, estimulo):
        """ Estimula la neurona con las seniales estimulo.
            Calcula z y a 
        """
        assert(len(estimulo) == len(self.weights))
        
        self.compute_z(estimulo)
        self.a = self.phi(self.z)
        return self.a

    def set_a(self, signal):
        """ Para una neurona de la capa input, setear su output """
        self.a = signal
        
    def print_neuron(self):
        """ Imprime neurona por pantalla, para debug """
        print "weights -> ", self.weights
        print "bias -> ", self.bias
        print "a -> ", self.a
        print "z -> ", self.z
        print "delta -> ", self.delta

