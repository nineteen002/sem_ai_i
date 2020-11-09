# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 20:00:22 2020

@author: Fernanda
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
        Sigma2_x and Sigma2_y values of sigma for each coordinate
    """
    def __init__(self,  i=0, x = 0, y = 0, s_x = 0, s_y = 0, f = 0):
        self.name = i
        self.x_coord = x
        self.y_coord = y
        self.sigma2_x = s_x
        self.sigma2_y = s_y
        self.evaluated = 0
        
    def initialize(self,xl, xu, yl, yu):
        """Initialized coordinates and sigma with random values
        Coordinate depending on the limits """
        x_r, y_r = round(random.uniform(0, 1),6), round(random.uniform(0, 1),6) #Generate random values between 0 and 1

        self.x_coord = round(xl + ((xu - xl) * x_r),6)  #Generate random x
        self.y_coord = round(yl + ((yu - yl) * y_r),6) #Generate random y
        self.sigma2_x = round(random.uniform(0, 3) * 0.2, 6 )
        self.sigma2_y = round(random.uniform(0, 3) * 0.2 , 6 )
        
    def print(self): #prints information on indiviual
        print("Individuo ", self.name, ":")
        print("Coordinates: (", self.x_coord, ", ", self.y_coord, ")")
        print("Sigmas x:", self.sigma2_x, "Sigma y", self.sigma2_y)

def z_function(x, y, f):
    """ Regresa el valor de la funcion evaluada en los punto x y y"""
    if(f == 1):
        return x * np.exp(-x**2 - y**2)
    if(f == 2):
        return (x - 2)**2 + (y -2)**2

def plot_graph(f, population, alg_used):
    """ Recieves x-axis and y-axis limits 
        Evaluates the function in steps of 100 between the limits
        Plots contour graph of the points found as well as the color bar that indicates depth
        Plots points (parents) of the algorthm
    """
    title = 'Function: ' + str(f) + alg_used
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
    for indv in population:
        plt.plot(indv.x_coord, indv.y_coord, color='red', marker='x', linewidth=0.30) #plot all of the points
        #print("Parent:", indv.x_coord, indv.y_coord, indv.evaluated)
    #PLOT BEST RESULT
    plt.plot(population[0].x_coord, population[0].y_coord, color='green', marker='o', linewidth=0.30) #plot all of the points
        
    plt.show()
    plt.clf()

def initialize(xl, xu, yl, yu):
    #generar padres con valores aliatorios y los agrega a la lista
    for i in range(mu):
        ind = Individuo(i)
        ind.initialize(xl, xu, yl, yu)
        population.append(ind)
        #population[i].print()
        
    return population

def recombine(parent1, parent2, i =0):
    #Crea nuevo hijo a partir del padre utilizando RECOMBINACION SEXUAL INTERMEDIA
    child_x = round((parent1.x_coord + parent2.x_coord )/2 , 6)
    child_y = round((parent1.y_coord + parent2.y_coord )/2 , 6)
    child_sx = round((parent1.sigma2_x + parent2.sigma2_x )/2, 6)
    child_sy = round((parent1.sigma2_y + parent2.sigma2_y )/2, 6)
    
    #CREA Y AGREGA EL HIJO A LA LISTA
    child = Individuo(i, child_x, child_y, child_sx, child_sy)
    children_pop.append(child)
    #child.print()
    
def mutacion(i):
    #GENERAR VALORES ALREATORIOS DE DISTRIBUCION NORMAL Y CALCULA EL VALOR DE X , Y 
    child = children_pop[i]
    
    #Random values from normal distribution
    x_r = round(np.random.normal(0, child.sigma2_x) , 6)
    y_r = round(np.random.normal(0, child.sigma2_y) , 6)
    
    #print("r = ", x_r, y_r)
    #Calculate new values por children
    children_pop[i].x_coord = children_pop[i].x_coord + x_r
    children_pop[i].y_coord = children_pop[i].y_coord + y_r
    

def evaluate_fitness(pop, f):
    for child in pop:
        child.evaluated = round(z_function(child.x_coord, child.y_coord, f),4) #Evaluar los puntos en la funcion
        
    #SORTS POPULATION, FOR EACH PARENT EVALUATED IN FUNCTION, estan de menor a mayor
    pop.sort(key=lambda x: x.evaluated, reverse = False)   
    """
    for child in pop:
        print("Child", child.name, "evaluated", child.evaluated) #SORTS POPULATION, FOR EACH PARENT EV
    """
    return pop
      

mu = 30
lamb = 150
dimension = 2
generations = 200

population = [] #population array
children_pop = []
fitness = []

def algorithm(f, algorithm, xl, xu, yl, yu):
    population = initialize(xl, xu, yl, yu)
    for g in range(generations):
        #CREATE CHILDREN
        for i in range(lamb):
            index = random.randint(0,mu-1) #chose random parents
            parent1 = population[index]
            parent2 = parent1
            
            while(parent1 == parent2): #chose different parents
                index = random.randint(0,mu-1) #chose random parent
                parent2 = population[index]
    
            recombine(parent1, parent2, i) #Generar hijo 
            mutacion(i) #
            
        
        if(algorithm == 1): #(m,lambda)
            tmp = evaluate_fitness(children_pop,f) #Orders children best to worst
            population = tmp[0:mu] #The children are now parents
            if(g%10 == 0): #Plot points every 10th generation
                str_used = "(m,lambda)"
                plot_graph(f, population, str_used)

            children_pop.clear()
        if(algorithm == 2): #(m + lambda)
            str_used = "(m + lambda)"
            tmp = population+children_pop #parents and children will be evaluated
            tmp = evaluate_fitness(tmp, f)#Evaluates parent and children
            population = tmp[0:mu] #Only select the best #mu individuals
            if(g%10 == 0):  #Plot points every 10th generation
                plot_graph(f, population, str_used)

            children_pop.clear()
    plt.pause(5)
    plt.close()
    return population[0]

x = sy.Symbol('x')
y = sy.Symbol('y')

print('----------------------------------------------------\n')
#First function
f_1 = x * sy.exp(-x**2 - y**2) 
#set limits
xl, xu = -2, 2
yl, yu = -2 ,2

print("Function: ", f_1, '\n')
best = algorithm(1,1, xl, xu, yl, yu)
print("using (M, lambda): x= ", best.x_coord, "y =", best.y_coord, "f(x,y) =", best.evaluated)
      
best = algorithm(1,2, xl, xu, yl, yu)
print("using (M + lambda): x= ", best.x_coord, "y =", best.y_coord, "f(x,y) =", best.evaluated)

print('----------------------------------------------------\n')
#Second function
f_2 = (x - 2)**2 + (y - 2)**2
#set limits
xl, xu = -3, 5
yl, yu = -3, 5

print("Function: ", f_2, '\n')
best = algorithm(2,1, xl, xu, yl, yu)
print("using (M, lambda): x= ", best.x_coord, "y =", best.y_coord, "f(x,y) =", best.evaluated)
      
best = algorithm(2,2, xl, xu, yl, yu)
print("using (M + lambda): x= ", best.x_coord, "y =", best.y_coord, "f(x,y) =", best.evaluated)



