# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 23:28:20 2020

@author: Fernanda
"""
import math
import numpy as np

def Ackley(chromosome):
    firstSum = 0.0
    secondSum = 0.0
    for c in chromosome:
        firstSum += c**2.0
        secondSum += math.cos(2.0*math.pi*c)
        n = float(len(chromosome))
    return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e

def Ack(x,y):
    sum1 = pow(x,2.0) + pow(y,2.0)
    sum2 = math.cos(2.0*math.pi*x) + math.cos(2.0*math.pi*y)
    n = 2.0
    return -20.0*math.exp(-0.2*math.sqrt(sum1/n)) - math.exp(sum2/n) + 20 + math.e

def Rastrigin( chromosome):
	"""F5 Rastrigin's function
	multimodal, symmetric, separable"""
	fitness = 10*len(chromosome)
	for i in range(len(chromosome)):
		fitness += chromosome[i]**2 - (10*math.cos(2*math.pi*chromosome[i]))
	return fitness

def Rastrig(x,y):
    return 20 +(pow(x,2.0) - (10*np.cos(2*np.pi*x))) + (pow(y,2.0) - (10*np.cos(2*np.pi*y)))

c = 1,1.5
print(Rastrigin(c))
print(Rastrig(1, 1.5))
