#!/usr/bin/python

from itertools import izip
import numpy as np
import random

from factivacion import *
from neurona import Neuron
from capa import Layer

class MLP(object):
    """ Implementa un perceptron multicapa """

    def __init__(self, neuronas_por_capa, phi=logistica(1), phi_prima = logistica_p(1), desvio=0.1):
        self.layers = []
        i = 0
        for t in neuronas_por_capa:
            if i == 0:
                l = Layer(t, 'input')
            elif i == len(neuronas_por_capa)-1:
                l = Layer(t, 'output')
            else:
                l = Layer(t, 'hidden')
            i = i + 1

            self.layers.append(l)

        # Conecto aleatoriamente las capas
        for l, lant in izip(self.layers[1:], self.layers[:-1]):
            l.random_connect(len(lant.neurons), desvio)
   
    def output(self):
        """ Devuelve la activacion de la ultima capa de la red """
        return self.layers[-1].get_output()

    def forward(self, x):
        """ Calcula y devuelve la prediccion de la red para el input x """
        assert(len(x) == len(self.layers[0].neurons)) # un input por neurona de la capa de input

        o = x
        for l in self.layers:
            o = l.propagate(o)

        return self.output()


    def compute_deltas(self, x, d):
        """ Computa los deltas de cada neurona, dado el input x y el valor deseado d """
        
        # Computar la prediccion de la red
        self.forward(x)

        next_layer = self.layers[-1]
        next_layer.compute_delta_output(d)

        for l in reversed(self.layers):
            #break
            if l.tipo == 'output':
                continue

            if l.tipo == 'input':
                break
            
            l.compute_delta_hidden(next_layer)
            next_layer = l

    def backprop(self, x, d, eta):
        # computar los deltas
        self.compute_deltas(x, d)

        # ajustar los pesos
        prev_layer = self.layers[0]
        for l in self.layers:
            if l.tipo == 'input':
                continue

            l.update_weights(eta, prev_layer)
            prev_layer = l

    def print_MLP(self):
        j = 0
        i = 0
        for l in self.layers:
            print "Layer ", i, "(", l.tipo,")",":"
            i = i + 1
            j = 0
            for n in l.neurons:
                print "Neurona", j, ":"
                n.print_neuron()
                j = j + 1
