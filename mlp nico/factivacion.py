""" Definicion de algunas funciones de activacion comunes """
import numpy as np

def logistica(a):
    return lambda x: 1 / (1 + np.exp(-a*x))

def logistica_p(a):
    return lambda x: a*logistica(a)(x)*(1 - logistica(a)(x))

