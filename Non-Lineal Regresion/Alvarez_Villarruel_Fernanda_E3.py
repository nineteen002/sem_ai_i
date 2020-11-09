# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:01:12 2020

@author: Fernanda
"""

import scipy.io
import sympy as sy
from sympy import exp
import numpy as np
import matplotlib.pyplot as plt
import random

data = scipy.io.loadmat('data_noLineal.mat')

xl, xu = 0, 10
yl, yu = 0, 20

w1_l, w1_u = 0.1, 0.7
w2_l, w2_u = 0.1, 0.7

X = data.get('X')
Y = data.get('Y')

n = (len(X))
w1 = 0.6
w2 = 0.4
 
def fm_b(w1, w2):
    """ Regresa el valor de la funcion evaluada en los punto m y b"""
    return  w1 * np.exp(w2*X)

def error(w1,w2):        
    return (1/(2*n))*sum(pow((Y - (w1 * np.exp(w2*X))),2))

class Individuo:
    """Class for individual
        Contrains name, coordinates in x and y
        z_coord <- Evaluated coordinates in function
        aptitud and probability to be choosen 
    """
    def __init__(self, w1_coord, w2_coord, i=0):
        self.name = i
        self.w1_coord = w1_coord
        self.w2_coord = w2_coord
        self.error = 0
        
        #Aptitud related
        self.aptitud = 0
        self.probability = 0
        
    def print(self): #prints information on indiviual
        print("Individuo ", self.name, ":")
        print("Coordinates: (", self.w1_coord, ", ", self.w2_coord, ") = ", self.error)
        print("Aptitud:", self.aptitud, "\t\tProbability:", self.probability)
        print('\n')

def plot_graph(f, best):
    """ Recieves x-axis and y-axis limits 
        Evaluates the function in steps of 100 between the limits
        Plots contour graph of the points found as well as the color bar that indicates depth
    """
    title = 'Function: ' + str(f) 
    plt.title(title)
    #Limits of the graph
        
    error_best = 99999999999 #Used to find best approximation
    best_indv =  Individuo(0,0)
    for indv in population:
        #Find best estimation
        if(indv.error < error_best):
            best_indv = indv
            error_best = indv.error
            w1_best = indv.w1_coord
            w2_best = indv.w2_coord
    
    if(error_best < best.error):
        plt.scatter(X, Y, color='red', marker='.', linewidths=0.5)
        x = np.linspace(xl, xu, n)
        y = fm_b(w1_best,w2_best)
        plt.plot(x, y, color='blue')
        plt.show()
        plt.clf()
    
    return best_indv

def initialize():
    """ Generates parent randomly """
    for i in range(N):
        w1_r, w2_r = round(random.uniform(0, 1),6), round(random.uniform(0, 1),6) #Generate random values between 0 and 1
        
        #generate random value of x and y
        x = round(w1_l + ((w1_u - w1_l) * w1_r),6) 
        y = round(w2_l + ((w2_u - w2_l) * w2_r),6)
        #Creates the individual with calculated values and adds it to population
        population.append(Individuo(x,y,i))

def aptitud(func):
    """ Calculates aptitud by evaluating the function in the points given
        Aptitud is only positive values
    """
    
    for indv in population: #recorre la poblacion
        indv.error = error(indv.w1_coord, indv.w2_coord) #evaluar funcion
        
        if (indv.error >= 0):  #Calcular aptitud
            indv.aptitud = (1/ (1 + indv.error))
        else: 
            indv.aptitud = (1+abs(indv.error))
        
def calculate_probability():
    total_aptitud = 0
    
    #Sum of aptituds    
    for indv in population:
        total_aptitud += indv.aptitud
    
    #Calculate individual probabilities
    for indv in population:
        indv.probability = indv.aptitud / total_aptitud 

def selection():
    #Use selection by ruleta
    r = round(random.uniform(0, 1),6) #Generate random number for selection
    psum = 0
    
    for indv in population: #Identify which individual was selected randomly and return it
        psum += indv.probability
        if(psum >= r):
            return indv
            break
    return population[-1]
    
def cruza(parent_1, parent_2, i):
    #Generate punto de cruza randomly
    pc = random.randint(0,D)
    
    #get parents coordinates ("dna")
    parent1_dna = [parent_1.w1_coord, parent_1.w2_coord]
    parent2_dna = [parent_2.w1_coord, parent_2.w2_coord]
    
    #generate childres dna from parents and pc value
    child1_dna = parent1_dna[0:pc] + parent2_dna[pc:]
    child2_dna = parent2_dna[0:pc] + parent1_dna[pc:]
    
    #ADD children to list of population
    children_pop.append(Individuo(child1_dna[0], child1_dna[1], i))
    children_pop.append(Individuo(child2_dna[0], child2_dna[1], i+1))

def mutar():
    #Generate mutations depending on probability slected
    pm = 0.002
    
    for indv in children_pop:
        w1_r, w2_r = round(random.uniform(0, 1),6), round(random.uniform(0, 1),6) #Generate random values between 0 and 1
        
        if(w1_r < pm): #If random value is less than probaility
            w1_b = round(random.uniform(0, 1),6) #Generate random values between 0 and 1       
            indv.w1_coord = round(w1_l + ((w1_u - w1_l) * w1_b),6) #Generates random coordinate
            
        if(w2_r < pm):
            w2_b = round(random.uniform(0, 1),6) #Generate random values between 0 and 1   
            indv.w2_coord = round(w2_l + ((w2_u - w2_l) * w2_b),6)

N = 200 #Size of population
D = 2 #Dimention 
Gen = 100 #Number of generations
population = [] #population array
children_pop = []


x = sy.Symbol('x')
y = sy.Symbol('y')

print('----------------------------------------------------\n')
#First function
#set limits

""" RUN GENETIC ALGORITHM """
initialize() #generates random population
best = population[0]
for g in range(Gen+1):
    aptitud(1) #calculates aptitude of each indv in population
    calculate_probability() #Estimates probability for each indv
        
    children_pop = [] #generate children population empty
    
    i = 0
    
    while(len(children_pop) < len(population)):
        #Select two parents
        r1 = selection()
        r2 = r1
        
        while(r1.name == r2.name): #Make sure its not the same parent
            r2 = selection()
        
        cruza(r1, r2, i) #generate children
        i += 2

    mutar() #Generate mutations
    
    best = plot_graph(1, best) #plot graph
    
    population = children_pop
   
print("Best approximation: (", best.w1_coord, ",", best.w2_coord, ") = ", best.error)
