import numpy as np
import math

class activator():

    def __init__(self,act_type=0):
        self.type = act_type

    def self_function(x):
        return x
    
    def self_derivative(x):
        return 1

    def sigmoid_function(x):
        return 1/(1+math.exp(-x))

    def sigmoid_derivative(x):
        t = 1/(1+math.exp(-x))
        return t*(1-t)

    act_function   =    {0: np.vectorize(self_function),
                        1: np.vectorize(sigmoid_function)}
    
    act_derivative =    {0: np.vectorize(self_derivative),
                        1: np.vectorize(sigmoid_derivative)}

    
    def function(self):
        return activator.act_function[self.type]

    def derivative(self):
        return activator.act_derivative[self.type]