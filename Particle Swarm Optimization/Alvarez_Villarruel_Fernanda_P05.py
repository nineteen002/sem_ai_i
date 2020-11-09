# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:17:30 2020

@author: Fernanda
"""

import sympy as sy
from sympy import exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import random

w = 0.6
c1, c2 = 2, 2

xl, xu = -3, 3
yl, yu = -3, 3

N = 30
G = 55
D = 2

particulas = []

def Griewank(x,y):
    res = ((pow(x,2))/4000.0) + ((pow(y,2))/4000.0)
    res_2 =  (np.cos(x/math.sqrt(1))) * (np.cos(y/math.sqrt(2)))
    return res-res_2 +1

def Sphere(x,y):
    return x**2 + y**2

def Ackley(x,y):
    sum1 = pow(x,2.0) + pow(y,2.0)
    sum2 = np.cos(2.0*np.pi*x) + np.cos(2.0*np.pi*y)
    n = 2.0
    return -20.0*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + math.e

def Rastrigin(x,y):
    return 20 +(pow(x,2.0) - (10*np.cos(2*np.pi*x))) + (pow(y,2.0) - (10*np.cos(2*np.pi*y)))

def f(x, y, opc):
    if(opc == 1):
        return Sphere(x,y)
    if(opc == 2):
        return Ackley(x, y)
    if(opc == 3):
        return Rastrigin(x, y)
    if(opc == 4):
        return Griewank(x, y)

def plot_graph(fun):
    """ Recieves x-axis and y-axis limits 
        Evaluates the function in steps of 100 between the limits
        Plots contour graph of the points found as well as the color bar that indicates depth
        Plots points (parents) of the algorthm
    """
    title = 'Function: ' + "0"
    plt.title(title)
    #Limits of the graph
    x = np.linspace(xl-10, xu+10, 300)
    y = np.linspace(yl-10, yu+10, 300)
    
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y, fun)
    #Plot graph and color bar
    plt.contour(X,Y,Z, levels=30)
    plt.colorbar()
    
    #plot population:
    for indv in particulas:
        plt.plot(indv.x, indv.y, color='red', marker='x', linewidth=0.30) #plot all of the points
        #print("Parent:", indv.x_coord, indv.y_coord, indv.evaluated)
    #PLOT BEST RESULT
    best = getBestPos(fun)
    plt.plot(best.x, best.y, color='green', marker='o', linewidth=0.30) #plot all of the points
    
    plt.show()
    plt.clf()

class Particula:
    """Class for individual
        Contrains name, coordinates in x and y, velocidad y mejor coordenada
        z_coord <- Evaluated coordinates in function
    """
    def __init__(self, i):
        self.name = i
        
        #generate random value of x and y
        x_r, y_r = round(random.uniform(0, 1),6), round(random.uniform(0, 1),6) #Generate random values between 0 and 1
        self.x = round(xl + ((xu - xl) * x_r),6)
        self.y = round(yl + ((yu - yl) * y_r),6)
        self.x_best = self.x
        self.y_best = self.y 
        self.v_x = np.random.randn(1)
        self.v_y = np.random.randn(1)
        
    def print(self):
        print("Name", self.name)
        print("x-coordinate:", self.x, "\ty-coordinate:", self.y)
        print("Velocidad: x", self.v_x, "y", self.v_y)
        print("Best coordinates: (", self.x_best, ",", self.y_best, ")")

def initialize():
    i = 1
    #Inicializar particulas, velocidades y mejores posiciones aleatoriamente
    for p in range(N):
        particulas.append(Particula(i))
        i += 1

def updateBestCoord(fun):  
    #para cada particula checa si la posicion actual es su mejor posicion
    for p in particulas:
        if(f(p.x, p.y, fun) < f(p.x_best, p.y_best, fun)):
            p.x_best = p.x
            p.y_best = p.y      

def getBestPos(fun):
    #Checa cada particula y regresa aquella con la mejor posicion
    best_p = particulas[0]
    for p in particulas:
        if(f(p.x, p.y, fun) < f(best_p.x, best_p.y, fun)):
            best_p = p
    return best_p

def calculateNewGen(x_g):
    #print("----------------------------------------------------")
    for p in particulas:
        r1, r2 = round(random.uniform(0, 1),6), round(random.uniform(0, 1),6)
        
        #print(r1, r2)
        #CALCULAR VELOCIDADES
        v_x = (w*p.v_x) + (r1*c1*(p.x_best - p.x)) + (r2*c2*(x_g.x - p.x))
        v_y = (w*p.v_y) + (r1*c1*(p.y_best - p.y)) + (r2*c2*(x_g.y - p.y))
        #print(p.v_x, v_x, p.v_y, v_y)
        p.v_x = v_x
        p.v_y = v_y
        #Calcular nuevas posiciones 
        p.x = p.x + p.v_x
        p.y = p.y + p.v_y

def algorithmEnjambre(fun):
    initialize()
            
    #por generacion
    for g in range(G):
        plot_graph(fun)
            
        updateBestCoord(fun)
        
        #for p in particulas:
            #p.print()
            #print("Evaluada:", f(p.x,p.y,fun))
        
        x_g = getBestPos(fun)
        #x_g.print()
        
        calculateNewGen(x_g)
    return x_g



#best = algorithmEnjambre(1)
#best.print()

#best = algorithmEnjambre(2)
#best.print()

best = algorithmEnjambre(3)
best.print()

best = algorithmEnjambre(4)
best.print()