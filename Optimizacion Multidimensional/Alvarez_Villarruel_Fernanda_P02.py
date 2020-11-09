# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:52:36 2020

@author: Fernanda Álvarez Villarruel
"""

import sympy as sy
from sympy import exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

#ax = plt.axes(projection="3d")

def z_function(x, y, f):
    """ Regresa el valor de la funcion evaluada en los punto x y y"""
    if(f == 1):
        return x * np.exp(-x**2 - y**2)
    if(f == 2):
        return (x - 2)**2 + (y -2)**2

#RANDOM SEARCH FUNCTIONS
def random_values(xl, yl, xu, yu):
    """ Calculates the values for x and y coordinates:
        
        x_r, y_r -> random value between 0 and 1 (including 0 and 1)
        x, y -> lower limit + (upper limit-lower limit) * random     
    """
    x_r, y_r = round(random.uniform(0, 1),6), round(random.uniform(0, 1),6)
    
    x = round(xl + ((xu - xl) * x_r),6)
    y = round(yl + ((yu - yl) * y_r),6)
    
    return x, y

def random_search(xl, yl, xu, yu, f, reps):
    """ Calculates random coordinates of x and y, a reps number of times
        Uses random_values function to calculate this coordinates and then 
        evaluates those coordinates if f(x,y).
        
        Checks if the f(x,y) is the lowest found and saves the value.
        Returns the x and y coordinates with the lowest evaluated result therefor finding a global minimum
    """
    z_best = 99999999999
    
    for i in range(reps):
        x, y = random_values(xl, yl, xu, yu)
        z = round(z_function(x, y, f), 7)
        
        #Checa si f(x,y) es menor al previo guardado, en caso que si lo guarda como en nuevo minimo
        if(z < z_best):
            z_best = z
            x_best = x
            y_best = y
            
    return x_best, y_best, z_best

def show_rs(x, y, f):
    "Shows results from random search"
    print("Random Search Algorithm Global Minimum Coordinates: ")
    print("x = ", x)
    print("y = ", y)
    print("F(x,y) = ", z_function(x, y, f))
    plt.plot(x, y, z_function(x, y, f), color='red', marker='x', linewidth=0.30)
    
#GRADIENT FUNCTIONS
def partial_derivative_x(f): 
    """Calcula derivada parcial en x de una funcion"""
    f_prime_x = sy.diff(f, x)
    return sy.lambdify((x, y), f_prime_x)
    
def partial_derivative_y(f): 
    """Calcula derivada parcial en y de una funcion"""
    f_prime_y = sy.Derivative(f, y).doit()
    return sy.lambdify((x, y), f_prime_y)

def calculate_gradient(x_init, y_init, f, iterations):    
    """ Recieves x and y initial coordinetes, a function and number of iterations
    h is a selected step value
    xi - h * Evaluate the partial derivatives for x and y for the points given.
    iterates calculations of x_i and y_i getting closer every time to the minimum 
    returns x and y coordinates for the minimm
    """
    f_partial_x = partial_derivative_x(f)
    f_partial_y = partial_derivative_y(f)
    
    x_i, y_i = x_init, y_init
    h = 0.1
    
    for i in range(iterations):
        x_next = x_i - h * f_partial_x(x_i, y_i)
        y_next = y_i - h * f_partial_y(x_i, y_i)

        x_i = round(x_next, 6)
        y_i = round(y_next, 6)
    
    return x_i, y_i

def show_gm(x, y, f):
    "Shows values of the Gradient Descent Method"
    print("Gradient Descent Method, Global Minimum Coordinates: ")
    print("x = ", x)
    print("y = ", y)
    print("F(x,y) = ", z_function(x, y, f))
    plt.plot(x, y, z_function(x, y, f), color='red', marker='x', linewidth=0.30)
    
        
def plot_graph(xl, xu, yl, yu, f, m):
    """ Recieves x-axis and y-axis limits 
        Evaluates the function in steps of 100 between the limits
        Plots contour graph of the points found as well as the color bar that indicates depth
    """
    title = 'Function: ' + str(f) + " using " + m
    plt.title(title)
    
    x = np.linspace(xl, xu, 100)
    y = np.linspace(yl, yu, 100)
    
    X, Y = np.meshgrid(x, y)
    Z = z_function(X, Y, f)
    
    #ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.contour(X,Y,Z, levels=45, cmap=cm.nipy_spectral )
    plt.colorbar()
    
    plt.show()
    plt.pause(5)
    plt.clf()

x = sy.Symbol('x')
y = sy.Symbol('y')

print('----------------------------------------------------\n')
#First function
f_1 = x * sy.exp(-x**2 - y**2) 
#set limits
xl, xu = -2, 2
yl, yu = -2 ,2

print("Function: ", f_1, '\n')
#USE RANDOM SEARCH METHOD
X, Y, Z = random_search(xl, yl, xu, yu, 1, 6000)
show_rs(X, Y, 1)
plot_graph(xl, xu, yl, yu, 1, "Random Search")

print("\n")
#USE GRADIENT METHOD
X, Y = calculate_gradient(-1, 1, f_1, 100)
show_gm(X, Y, 1)
plot_graph(xl, xu, yl, yu, 1, "Gradient Descent")


print('----------------------------------------------------\n')
#Second function
f_2 = (x - 2)**2 + (y -2)**2
#set limits
xl, xu = 0, 4
yl, yu = 0, 4

print("Function: ", f_2, '\n')
#USE RANDOM SEARCH METHOD
X, Y, Z = random_search(xl, yl, xu, yu, 2, 6000)
show_rs(X, Y, 1)
plot_graph(xl, xu, yl, yu, 2, "Random Search")

print("\n")
#USE GRADIENT METHOD
X, Y = calculate_gradient(-1, 1, f_2, 100)
show_gm(X, Y, 1)
plot_graph(xl, xu, yl, yu, 2, "Gradient Descent")


