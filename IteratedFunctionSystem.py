#!/usr/bin/env python3 

# IteratedFunctionSystem.py 
# Author: Alexander Lind 

import numpy as np

# Class implementing a 2D affine transformation on the form M * x + v 
class AffineTransformation: 

    # Initialise a zero transformation matrix and translation vector 
    def __init__(self): 
        self.matrix = np.zeros((2, 2))
        self.vector = np.zeros(2)

    def initialise(self, matrix, vector): 
        self.matrix = matrix
        self.vector = vector 

    def generateRandomAffineTransformation(self): 
        matrix = np.random.rand(2, 2) 
        determinant = np.linalg.det(matrix) 
        if (determinant != 0): 
            matrix = matrix / abs(determinant)**(1.0/2.0)
        self.matrix = matrix 
        self.vector = np.zeros(2)

    # Apply transformation 
    def evaluate(self, x, y): 
        xp = self.matrix[0, 0] * x + self.matrix[0, 1] * y + self.vector[0] 
        yp = self.matrix[1, 0] * x + self.matrix[1, 1] * y + self.vector[1] 
        return (xp, yp)

    def updateParameters(self, data, # 2D input vector 
                               loss, # 2D loss vector 
                               eta): # Learning rate 

        #print('data = ' + str(data))
        #print('loss = ' + str(loss))

        self.matrix[0, 0] -= eta * loss[0] * data[0] # a 
        self.matrix[0, 1] -= eta * loss[0] * data[1] # b 
        self.matrix[1, 0] -= eta * loss[1] * data[0] # c 
        self.matrix[1, 1] -= eta * loss[1] * data[1] # d 

        self.vector[0] -= eta * loss[0] # e 
        self.vector[1] -= eta * loss[1] # f 

    def output(self): 
        print('[' + str(self.matrix[0, 0]) + ', ' + str(self.matrix[0, 1]) + ']  [' + str(self.vector[0]) + ']')
        print('[' + str(self.matrix[1, 0]) + ', ' + str(self.matrix[1, 1]) + ']  [' + str(self.vector[1]) + ']')

# Class implementing an iterated function system 
class IteratedFunctionSystem: 

    def __init__(self): 
        self.transformations = [] 
        self.probabilities = [] 

    # Add a given affine transformation to the IFS 
    def addAffineTransformation(self, t = None, # Transformation 
                                      m = None, # Matrix 
                                      v = None, # Vector 
                                      p = 0):   # Probability 

        if t is None: 
            if m is None: 
                raise ArgumentError('Please supply either an affine transformation or a matrix and a vector')
            else: 
                transformation = AffineTransformation() 
                transformation.initialise(m, v) 
                self.transformations.append(transformation)
        else: 
            self.transformations.append(t)

        self.probabilities.append(p)

    # Add one (or more) random affine transformation(s) to the IFS 
    def addRandomTransformations(self, n = 1): 
        for i in range(n): 
            transformation = AffineTransformation() 
            transformation.generateRandomAffineTransformation() 
            self.transformations.append(transformation)
            self.probabilities.append(1.0)

    # Evaluate the IFS at input point (x, y) 
    def evaluate(self, x, y): 

        n = len(self.transformations) 

        # Normalise probabilities to 1 
        psum = sum(self.probabilities)
        if (psum != 1.0): 
            for i in range(len(self.probabilities)): 
                self.probabilities[i] /= psum 

        # Choose one of the affine transformations with their respective weighted probabilities 
        transformChoice = np.random.choice(n, p = self.probabilities) 
        
        # Evaluate the chosen affine transformation on the input 
        result = self.transformations[transformChoice].evaluate(x, y)
        return (transformChoice, result[0], result[1])

    def updateParameters(self, nf,   # The affine transformation function to update 
                               data, # 2D input vector 
                               loss, # 2D loss vector 
                               eta): # Learning rate 

        self.transformations[nf].updateParameters(data, loss, eta) 

    def output(self): 
        for i in range(len(self.transformations)): 
            self.transformations[i].output()
            print('Probability = ' + str(self.probabilities[i]) + '\n')

# The black spleenworth fern fractal 
blackSpleenwortFern = IteratedFunctionSystem()
blackSpleenwortFern.addAffineTransformation(m = np.matrix([[ 0,     0],    [ 0,    0.16]]), v = [0, 0],    p = 0.01) 
blackSpleenwortFern.addAffineTransformation(m = np.matrix([[ 0.85,  0.04], [-0.04, 0.85]]), v = [0, 1.6],  p = 0.85) 
blackSpleenwortFern.addAffineTransformation(m = np.matrix([[ 0.20, -0.26], [ 0.23, 0.22]]), v = [0, 1.6],  p = 0.07) 
blackSpleenwortFern.addAffineTransformation(m = np.matrix([[-0.15,  0.28], [ 0.26, 0.24]]), v = [0, 0.44], p = 0.07) 

# The Sierpinsky Gasket 
sierpinskyGasket = IteratedFunctionSystem()
sierpinskyGasket.addAffineTransformation(m = np.matrix([[0.5, 0], [0, 0.5]]), v = [0,    0],            p = 1/3) 
sierpinskyGasket.addAffineTransformation(m = np.matrix([[0.5, 0], [0, 0.5]]), v = [0.5,  0],            p = 1/3) 
sierpinskyGasket.addAffineTransformation(m = np.matrix([[0.5, 0], [0, 0.5]]), v = [0.25, np.sqrt(3)/4], p = 1/3) 

# The Pythagorean Tree 
angle = 90.0 # NumPy apparently uses degrees and not radians 
pythagoreanTree = IteratedFunctionSystem()
pythagoreanTree.addAffineTransformation(m = np.matrix([[np.cos(angle)**2, -np.cos(angle)*np.sin(angle)], [np.cos(angle)*np.sin(angle), np.cos(angle)**2]]), v = [0, 1], p = 1/3) 
pythagoreanTree.addAffineTransformation(m = np.matrix([[np.sin(angle)**2, np.cos(angle)*np.sin(angle)], [-np.cos(angle)*np.sin(angle), np.sin(angle)**2]]), v = [np.cos(angle)**2, 1+np.cos(angle)*np.sin(angle)], p = 1/3) 
pythagoreanTree.addAffineTransformation(m = np.matrix([[1, 0], [0, 1]]), v = [0, 0], p = 1/3) 

# The Heighway Dragon 
heighwayDragon = IteratedFunctionSystem()
heighwayDragon.addAffineTransformation(m = np.matrix([[ 0.5, -0.5], [0.5,  0.5]]), v = [0, 0], p = 0.5) 
heighwayDragon.addAffineTransformation(m = np.matrix([[-0.5, -0.5], [0.5, -0.5]]), v = [1, 0], p = 0.5) 

# The Koch Curve 
kochCurve = IteratedFunctionSystem()
kochCurve.addAffineTransformation(m = np.matrix([[1/3, 0], [0,  1/3]]), v = [0, 0], p = 0.25) 
kochCurve.addAffineTransformation(m = np.matrix([[1/6, -np.sqrt(3)/6], [np.sqrt(3)/6,  1/6]]), v = [1/3, 0], p = 0.25) 
kochCurve.addAffineTransformation(m = np.matrix([[1/6, np.sqrt(3)/6], [-np.sqrt(3)/6,  1/6]]), v = [1/2, np.sqrt(3)/6], p = 0.25) 
kochCurve.addAffineTransformation(m = np.matrix([[1/3, 0], [0,  1/3]]), v = [2/3, 0], p = 0.25) 
