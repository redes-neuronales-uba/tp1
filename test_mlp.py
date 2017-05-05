#!/usr/bin/python

from mlp import *
import random

def logistica(a):
    return lambda x: 1 / (1 + np.exp(-a*x))


def logistica_p(a):
    return lambda x: a*logistica(a)(x)*(1 - logistica(a)(x))


def test_create_random_matrix():
    print create_random_matrix(2, 5, 0.2)


def test_init():
    n = MLP([2, 3, 2, 3])
    n.print_wyb()


def test_set_wyb():
    n = MLP([2, 2, 2])
    
    W = []
    B = []

    W.append(np.array([[0.15, 0.20], [0.25, 0.30]]))
    W.append(np.array([[0.40, 0.45], [0.50, 0.55]]))

    B.append(np.array([0.35, 0.35]))
    B.append(np.array([0.60, 0.60]))

    n.set_wyb(W, B)
    
    n.print_wyb()


def test_feedForward():
    x = [1, 0]
    n = MLP([2, 3, 2, 3])
    print n.feedForward(x)


def test_haykin_xor():
    a = 50
    n = MLP([2,2,1], phi=logistica(a), phi_prima=logistica_p(a))
    W = []
    B = []
    
    W.append(np.array([[1, 1], [1, 1]]))
    W.append(np.array([-2, 1]))
    
    B.append(np.array([-1.5, -0.5]))
    B.append(np.array([-0.5]))
    
    n.set_wyb(W, B)

    print "[1,1] -> ", n.feedForward([1,1])
    print "[1,0] -> ", n.feedForward([1,0])
    print "[0,1] -> ", n.feedForward([0,1])
    print "[0,0] -> ", n.feedForward([0,0])


def test_backprop():
    n = MLP([2, 3, 2, 3])
    n.backprop([1, 2], [1, 2, 3])


def test_mazzur():
    a=1
    n = MLP([2, 2, 2], eta=0.5, phi=logistica(a), phi_prima=logistica_p(a))
    
    W = []
    B = []

    W.append(np.array([[0.15, 0.20], [0.25, 0.30]]))
    W.append(np.array([[0.40, 0.45], [0.50, 0.55]]))

    B.append(np.array([0.35, 0.35]))
    B.append(np.array([0.60, 0.60]))
    
    n.set_wyb(W, B)
    
    n.backprop([0.05, 0.10], [0.01, 0.99])
    n.print_wyb()



lset_xor = [
       ([1,1], [-1]),
       ([1,-1], [1]),
       ([-1,1], [1]),
       ([-1,-1], [-1]) 
       ]

lset_and = [
       ([1,1], [1]),
       ([1,0], [0]),
       ([0,1], [0]),
       ([0,0], [0]) 
       ]

lset_or = [
       ([1,1], [1]),
       ([1,0], [1]),
       ([0,1], [1]),
       ([0,0], [0]) 
       ]

        
def test_xor():    
    a = -1
    n = MLP([2,2,1], eta=0.1, phi=logistica(a), phi_prima=logistica_p(a))

    it = 10
    for i in range(it):
        x, d = random.choice(lset_xor)
        print n.backprop(x, d)

    for case in lset_xor:
        print case[0], "-> ", n.feedForward(case[0]), " expected: ", case[1]



def test_or():    
    a = 0.01
    n = MLP([2,2,1], eta=1, phi=logistica(a), phi_prima=logistica_p(a))

    it = 500
    for i in range(it):
        x, d = random.choice(lset_or)
        n.backprop(x, d)

    for case in lset_or:
        print case[0], "-> ", n.feedForward(case[0]), " expected: ", case[1]


#TESTS
#test_create_random_matrix()
#test_init()
#test_set_wyb()

#Tests para feedForward
#test_feedForward()
#test_haykin_xor()

#Tests para backprop
#test_backprop()
#test_mazzur()
#test_xor()
test_or()

