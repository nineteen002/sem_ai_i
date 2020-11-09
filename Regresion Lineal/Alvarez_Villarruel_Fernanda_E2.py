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
from matplotlib import cm
import random

data = scipy.io.loadmat('data_noLineal.mat')

xl, xu = -5, 16
yl, yu = -2, 7

ml, mu = 0, 2
bl, bu = -0, 1

X = data.get('X')
Y = data.get('Y')

n = (len(X))
m = 0.8
b = 0.5
 
def fm_b(m, b):
    """ Regresa el valor de la funcion evaluada en los punto m y b"""
    return  m*X +b

def error(m,b):        
    return (1/(2*n))*sum((Y - (m*X +b))**2)
    
def error(m,b):        
    return (1/(2*n))*sum((Y - (m*X +b))**2)

#RANDOM SEARCH FUNCTIONS
def random_values(ml, bl, mu, bu):
    """ Calculates the values for x and y coordinates:
        
        x_r, y_r -> random value between 0 and 1 (including 0 and 1)
        x, y -> lower limit + (upper limit-lower limit) * random     
    """
    m_r, b_r = round(random.uniform(0, 1),6), round(random.uniform(0, 1),6)
    
    m = round(ml + ((mu - ml) * m_r),6)
    b = round(bl + ((bu - bl) * b_r),6)
    
    return m, b

def random_search(reps=1):
    """ Calculates random coordinates of x and y, a reps number of times
        Uses random_values function to calculate this coordinates and then 
        evaluates those coordinates if f(x,y).
        
        Checks if the f(x,y) is the lowest found and saves the value.
        Returns the x and y coordinates with the lowest evaluated result therefor finding a global minimum
    """
    best_error = 99999999999
    
    for i in range(reps):
        m, b = random_values(ml, bl, mu, bu)
        
        current_error = error(m, b)
        
        #Checa si f(x,y) es menor al previo guardado, en caso que si lo guarda como en nuevo minimo
        if(current_error < best_error):
            m_best = m
            b_best = b
            best_error = current_error
            plot_graph(m, b)
            #print(i, "m=", m, "b=", b, "Error", best_error)
            
    return m_best, b_best, best_error

 
#GRADIENT FUNCTIONS
def partial_derivative_m(m, b): 
    """Calcula derivada parcial en x de una funcion"""  
    suma = 0
    for i in range(n):
        suma -= (1/n) * X[i] * (Y[i] - (b + m*X[i]))
        
    return float(suma)
    
def partial_derivative_b(m, b): 
    """Calcula derivada parcial en y de una funcion"""
    suma = 0
    for i in range(n):
        suma -= (1/n) * (Y[i] - (b + m*X[i]))
    
    return float(suma)

def calculate_gradient(m_init, b_init, iterations):    
    """ Recieves x and y initial coordinetes, a function and number of iterations
    h is a selected step value
    xi - h * Evaluate the partial derivatives for x and y for the points given.
    iterates calculations of x_i and y_i getting closer every time to the minimum 
    returns x and y coordinates for the minimm
    """
    
    m_i, b_i = m_init, b_init
    h = 0.015
    
    for i in range(iterations):
        f_partial_m = partial_derivative_m(m_i, b_i)
        f_partial_b = partial_derivative_b(m_i, b_i)
        
        m_next = m_i - (h * f_partial_m)
        b_next = b_i - (h * f_partial_b)
                   
        m_i = round(m_next, 6)
        b_i = round(b_next, 6)
            
        if(i % 100 == 0):
            plot_graph(m_i, b_i)
        
    return m_i, b_i

def show_gm(x, y, f):
    "Shows values of the Gradient Descent Method"
    print("Gradient Descent Method, Global Minimum Coordinates: ")
    print("x = ", x)
    print("y = ", y)
    print("F(x,y) = ", z_function(x, y, f))
    plt.plot(x, y, z_function(x, y, f), color='red', marker='x', linewidth=0.30)
    
        
def plot_graph(m, b):
    """ Recieves x-axis and y-axis limits 
        Evaluates the function in steps of 100 between the limits
        Plots contour graph of the points found as well as the color bar that indicates depth
    """
    title = 'Function: '
    plt.title(title)
    
    x = np.linspace(xl, xu, n)
    
    plt.scatter(X, Y, color='red', marker='.', linewidths=0.5)
    plt.plot(x, fm_b(m,b), color='blue')
    
    plt.show()
    plt.clf()


print("Random Search Results:")
plot_graph(m, b)

m, b, e = random_search(10000)
print("m=", m, "b=", b, "Error", e)

#plots graph
plt.scatter(X, Y, color='red', marker='.', linewidths=0.5)
x = np.linspace(xl, xu, n)
plt.plot(x, fm_b(m,b), color='green')
plt.show()
plt.pause(5)
plt.close()

print("Gradient Descent Method Results: ")

m = 0.8
b = 0.5
 
m, b = calculate_gradient(m, b, 1000)
print("m=", m, "b=", b, "Error", error(m, b))
#plots graph
plt.scatter(X, Y, color='red', marker='.', linewidths=0.5)
plt.plot(x, fm_b(m,b), color='green')


