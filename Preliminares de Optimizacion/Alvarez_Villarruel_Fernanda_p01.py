# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:19:01 2020

@author: Fernanda Álvarez Villarruel
"""

import sympy as sy
from sympy import sin, cos
import matplotlib.pyplot as plt
import numpy as np

def func_obj(x, i):
    """ 
    Evaluates a function at x point and return the evaluate value of f(x)
    i: indicates which function will be used to evaluate x
    x: point at which the function is evaluated
    """
    if(i == 1):
        return x**4 + 5*x**3 +4*x**2 - 4*x + 1
    if(i == 2):
        return np.sin(2*x)
    if(i == 3):
        return np.sin(x) + (x * np.cos(x))

def first_derivative(f):
    """ Recibe la funcion y regresa la primera derivdada """
    f_prime= sy.Derivative(f, x).doit()
    f_prime = sy.lambdify(x, f_prime)
    return f_prime
    
def second_derivative(f):
    """ Recibe la funcion y regresa la segunda derivdada """
    f_prime= sy.Derivative(f, x).doit()
    
    f_2prime = sy.Derivative(f_prime, x).doit()
    f_2prime = sy.lambdify(x, f_2prime)
    return f_2prime

def plot_graph(x_start, x_end, x_step, y, zeros, f):
    """
    Parameters
    ----------
    x_start : int
        x-axis lower limit
    x_end : int
        x-axis upper limit
    x_step : int
        steps between each point that will be plotted
    y : index
        Chooses the function that will be used
    zeros : list
        Its a list contains the zeros found, used to plot the points in the graph
    f : str
        string of the function

    Returns
    -------
    None.

    """
    x = np.linspace(x_start, x_end, x_step)
    
    #Plot function
    title = 'Function: ' + str(f)
    plt.title(title)
    plt.plot(x, func_obj(x,y), color='black')
    
    #Plot zeros
    for zero in zeros:
        plt.plot(zero, func_obj(zero,y), color='blue', marker='o')
    
    plt.show()
    plt.pause(3)
    plt.clf()

def newton_raphson(f, initial, iterations):
    """ 
   This function uses the method of newton raphson to approximate a zero, using the derivatives of a function 
   
    Given a function -> f
    An initial value -> initial
    And the numnber of iterations
    
    returns: zero found
    """
    f_prime= first_derivative(f)
    f_2prime = second_derivative(f)
    
    x_i = initial
    for i in range(iterations):
        x_next = x_i - (f_prime(x_i) / f_2prime(x_i))
        x_i = x_next
    return x_i

def is_maximum(zero, f):
    """
     Calculates if the zeros found are maximum or minimum points by evaluating it in the second derivative
     If the evaluated zero in the second derivative is postive then its categorized as a Minimum
     If the evaluated zero in the second derivative is negative then its categorized as a Maximum
    """
    f_2prime = second_derivative(f)
    
    if(f_2prime(zero) > 0):
        print("Zero found at x =", zero, "is a Minimum")    
    else:
        print("Zero found at x =", zero, "is a Maximum")  


#MAKE CALCULATIONS FOR EACH FUNCTION

x = sy.Symbol('x')
print('----------------------------------------------------\n')
#First function
f1 = x**4 + 5*x**3 +4*x**2 - 4*x + 1

#Write limits of x-axis and step
x_l = -4
x_u = 1
step = 1500

#CALCULATES ZEROS
f1_zeros = [newton_raphson(f1, -3, 20), newton_raphson(f1, -1, 20), newton_raphson(f1, 0, 20)]

print("Function: ", f1, '\n')

#CHECKS IF THE ZEROS FOUND ARE MAX OR MINS
for zero in f1_zeros:
    is_maximum(round(zero,4), f1)
    
#plots graph and zeros
plot_graph(x_l, x_u, step, 1, f1_zeros, f1)
print('----------------------------------------------------\n')

#Second function
f2 = sin(2*x)
#Write limits of x-axis and step
x_l = -4
x_u = 4
step = 1500

#CALCULATES ZEROS
f2_zeros = [newton_raphson(f2, -4, 20), newton_raphson(f2, -2, 20), newton_raphson(f2, -0.5, 20), newton_raphson(f2, 1, 20), 
            newton_raphson(f2, 2, 20), newton_raphson(f2, 3.5, 20)]

print("Function: ", f2, '\n')

#CHECKS IF THE ZEROS FOUND ARE MAX OR MINS
for zero in f2_zeros:
    is_maximum(round(zero,4), f2)

#plots graph and zeros
plot_graph(x_l, x_u, step, 2, f2_zeros, f2)
print('----------------------------------------------------\n')

#Third function
f3 = sin(x) + (x * cos(x))
#Write limits of x-axis and step
x_l = -5
x_u = 5
step = 1500

#CALCULATES ZEROS
f3_zeros = [newton_raphson(f3, -3, 20), newton_raphson(f3, -1, 20), newton_raphson(f3, 1, 20), newton_raphson(f3, 4, 20)]

print("Function: ", f3, '\n')

#CHECKS IF THE ZEROS FOUND ARE MAX OR MINS
for zero in f3_zeros:
    is_maximum(round(zero,4), f3)

#plots graph and zeros
plot_graph(x_l, x_u, step, 3, f3_zeros, f3)
print('----------------------------------------------------\n')

