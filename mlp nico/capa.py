#!/usr/bin/python

from neurona import Neuron

from itertools import izip
import random
import numpy as np

class Layer(object):
    """ Implementa una capa de una red neuronal multicapa """

    def __init__(self, cantidad, tipo='hidden'):
        """ Constructor de Layer. Recibe una cantidad de neuronas y el tipo de capa
        (input, hiddden o output) """
        assert(cantidad > 0) # al menos una neurona
        assert(tipo in ['input', 'hidden', 'output'])
        
        self.tipo = tipo

        # creo cant neuronas default
        self.neurons = [Neuron() for i in range(cantidad)]
    
    def random_connect(self, cant, desvio):
        """ Agrega *cant* pesos y un bias a cada neurona de la capa. Los pesos y el bias son una 
        variable aleatroria con distribucion U[-desvio, desvio]"""
        for n in self.neurons:
            n.weights = [random.uniform(-desvio, desvio) for i in range(cant)]  # modificarlo para usar una varianza de m ala 1/2 
            n.bias = random.uniform(-desvio, desvio)                            # donde m es la cantidad de conexiones de una neurona

    def set_input(self, x):
        assert(self.tipo == 'input') # tiene que ser capa de input
        assert(len(self.neurons) == len(x)) # cada neurona representa un input

        for n, xi in izip(self.neurons, x):
            n.a = xi

    def get_output(self):
        return [n.a for n in self.neurons]

    def get_output_j(self, i):
        return self.neurons[i].a
    
    def propagate(self, x):
        """ Propaga la senial x por la capa.
            Si es una capa input, seteo los input y devuelvo oca
            Devuelve el output de esta capa """

        
        if self.tipo == 'input':
            assert(len(self.neurons) == len(x)) # un input por neurona de la capa input
            self.set_input(x)
            return x

        else:
            assert(len(self.neurons[0].weights) == len(x)) # una senial por cada peso en las neuronas de la capa
            return [n.estimular(x) for n in self.neurons]
    
    def get_error(self, d):
        """ Devuelve el error en una capa, dado un output d deseado """        
        assert(len(d) == len(self.neurons))
        return [n.a - di for n, di in izip(self.neurons, d)]
    
    def delta_vector(self):
        return [n.delta for n in self.neurons]

    def compute_delta_output(self, d):
        """ Computa el vector delta asumiendo que la neurona es de una capa output 
        d es el output deseado. Devuelve el vector delta calculado y actualiza el valor de delta
        en las neuronas"""
        #print d
        #print len(self.neurons)
        assert(len(d) == len(self.neurons))
        assert(self.tipo == 'output')
        
        # calculo y devuelvo el delta para esta capa
        
        delta = []
        for n, di in izip(self.neurons, d):
            n.delta = (n.a - di) * n.phi_prima(n.z)
            delta.insert(len(delta), n.delta)

        return delta

    def compute_delta_hidden(self, next_layer):
        """ Computa el delta asumiendo que la neurona es de una capa hidden
        d es el output deseado. Actualiza los valores del delta de las neuronas de la capa.
        Devuelve el vector delta computado"""
        assert(self.tipo == 'hidden')
        
        delta = []
        k = 0
        for n in self.neurons:
            wmk = [m.weights[k] for m in next_layer.neurons]
            next_layer_delta_vector = [nln.delta for nln in next_layer.neurons]
            delta_k = np.dot(next_layer.delta_vector(), wmk) * n.phi_prima(n.z)
            n.delta = delta_k
            delta.insert(len(delta), delta_k)
            k = k + 1 

        return delta

    def update_weights(self, eta, prev_layer):
        for n in self.neurons:
            #ajusto el bias
            n.bias = n.bias - eta * n.delta

            # ajusto los pesos 
            j = 0
            for w in n.weights:
                n.weights[j] = w - eta * n.delta * prev_layer.get_output_j(j)
                j = j + 1

