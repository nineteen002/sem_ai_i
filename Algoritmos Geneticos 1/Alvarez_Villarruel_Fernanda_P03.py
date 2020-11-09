# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sympy as sy
from sympy import exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

class Individuo:
    """Class for individual
        Contrains name, coordinates in x and y
        z_coord <- Evaluated coordinates in function
        aptitud and probability to be choosen 
    """
    def __init__(self, x_coord, y_coord, i=0):
        self.name = i
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.z_coord = 0
        
        #Aptitud related
        self.aptitud = 0
        self.probability = 0
        
    def print(self): #prints information on indiviual
        print("Individuo ", self.name, ":")
        print("Coordinates: (", self.x_coord, ", ", self.y_coord, ") = ", self.z_coord)
        print("Aptitud:", self.aptitud, "\t\tProbability:", self.probability)
        print('\n')

def z_function(x, y, f):
    """ Regresa el valor de la funcion evaluada en los punto x y y"""
    if(f == 1):
        return x * np.exp(-x**2 - y**2)
    if(f == 2):
        return (x - 2)**2 + (y -2)**2

def plot_graph(f):
    """ Recieves x-axis and y-axis limits 
        Evaluates the function in steps of 100 between the limits
        Plots contour graph of the points found as well as the color bar that indicates depth
    """
    title = 'Function: ' + str(f) 
    plt.title(title)
    #Limits of the graph
    x = np.linspace(xl, xu, 100)
    y = np.linspace(yl, yu, 100)
    
    X, Y = np.meshgrid(x, y)
    Z = z_function(X, Y, f)
    #Plot graph and color bar
    plt.contour(X,Y,Z, levels=45, cmap=cm.nipy_spectral )
    plt.colorbar()
    
    #plot population:
    z_best = 99999999999 #Used to find best approximation
    best_indv =  Individuo(0,0)
    for indv in population:
        #Find best estimation
        if(indv.z_coord < z_best):
            best_indv = indv
            z_best = indv.z_coord
            x_best = indv.x_coord
            y_best = indv.y_coord
        plt.plot(indv.x_coord, indv.y_coord, color='red', marker='x', linewidth=0.30) #plot all of the points
        
    plt.plot(x_best, y_best , color='green', marker='o', linewidth=0.30) #plot best point found
    
    plt.show()
    plt.clf()
    return best_indv

def initialize():
    """ Generates parent randomly """
    for i in range(N):
        x_r, y_r = round(random.uniform(0, 1),6), round(random.uniform(0, 1),6) #Generate random values between 0 and 1
        
        #generate random value of x and y
        x = round(xl + ((xu - xl) * x_r),6) 
        y = round(yl + ((yu - yl) * y_r),6)
        #Creates the individual with calculated values and adds it to population
        population.append(Individuo(x,y,i))

def aptitud(func):
    """ Calculates aptitud by evaluating the function in the points given
        Aptitud is only positive values
    """
    
    for indv in population: #recorre la poblacion
        indv.z_coord = round(z_function(indv.x_coord, indv.y_coord, func), 6) #evaluar funcion
        
        if (indv.z_coord >= 0):  #Calcular aptitud
            indv.aptitud = round(1/ (1 + indv.z_coord),4)
        else: 
            indv.aptitud = round(1+abs(indv.z_coord),4)

def calculate_probability():
    total_aptitud = 0
    
    #Sum of aptituds    
    for indv in population:
        total_aptitud += indv.aptitud
    
    #Calculate individual probabilities
    for indv in population:
        indv.probability = round( indv.aptitud / total_aptitud ,6)

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
    parent1_dna = [parent_1.x_coord, parent_1.y_coord]
    parent2_dna = [parent_2.x_coord, parent_2.y_coord]
    
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
        x_r, y_r = round(random.uniform(0, 1),6), round(random.uniform(0, 1),6) #Generate random values between 0 and 1
        
        if(x_r < pm): #If random value is less than probaility
            x_b = round(random.uniform(0, 1),6) #Generate random values between 0 and 1       
            indv.x_coord = round(xl + ((xu - xl) * x_b),6) #Generates random coordinate
            
        if(y_r < pm):
            y_b = round(random.uniform(0, 1),6) #Generate random values between 0 and 1   
            indv.y_coord = round(yl + ((yu - yl) * y_b),6)

N = 100 #Size of population
D = 2 #Dimention 
Gen = 100 #Number of generations
population = [] #population array
children_pop = []


x = sy.Symbol('x')
y = sy.Symbol('y')

print('----------------------------------------------------\n')
#First function
f_1 = x * sy.exp(-x**2 - y**2) 
#set limits
xl, xu = -2, 2
yl, yu = -2 ,2

print("Function: ", f_1, '\n')

""" RUN GENETIC ALGORITHM """
initialize() #generates random population

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
    
    best = plot_graph(1) #plot graph
    
    population = children_pop
    
print("Best approximation: (", best.x_coord, ",", best.y_coord, ") = ", best.z_coord)
aptitud(1)


population = [] #population array
children_pop = []
print('----------------------------------------------------\n')
#Second function
f_2 = (x - 2)**2 + (y - 2)**2
#set limits
xl, xu = -3, 5
yl, yu = -3, 5

print("Function: ", f_1, '\n')

initialize() #generates random population

for g in range(Gen+1):
    aptitud(2) #calculates aptitude of each indv in population
    calculate_probability()
        
    children_pop = [] #generate children population empty
    
    i = 0
    
    while(len(children_pop) < len(population)):
        #Select two parents
        r1 = selection()
        r2 = r1
        
        while(r1.name == r2.name): #Make sure its not the same parent
            r2 = selection()
        
        cruza(r1, r2, i)
        i += 2

    mutar()
    best = plot_graph(2)
    population = children_pop
print("Best approximation: (", best.x_coord, ",", best.y_coord, ") = ", best.z_coord)
aptitud(2)
