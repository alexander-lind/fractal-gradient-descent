#!/usr/bin/env python3 

# FractalImage.py 
# Author: Alexander Lind 
# Using iterated function systems to generate fractals 
# In this case we consider the black spleenwort fern 

import numpy as np
from matplotlib import pyplot as plt

import IteratedFunctionSystem as IFS

# Generate fractal image data 
def generateFractalImage(iterations = 100000, a = 0, b = 0): 

    # We start at (a, b) 
    x = [a] 
    y = [b] 

    # Generate fractal 
    for i in range(1, iterations): 

        i, xp, yp = IFS.blackSpleenwortFern.evaluate(x[i - 1], y[i - 1])
        #i, xp, yp = IFS.sierpinskyGasket.evaluate(x[i - 1], y[i - 1])
        #i, xp, yp = IFS.pythagoreanTree.evaluate(x[i - 1], y[i - 1])
        #i, xp, yp = IFS.heighwayDragon.evaluate(x[i - 1], y[i - 1])
        #i, xp, yp = IFS.kochCurve.evaluate(x[i - 1], y[i - 1])

        x.append(xp) 
        y.append(yp) 

    return (x, y) 

if __name__ == "__main__":

    # The IFS is a fixed attractor for the fractal, so it doesn't matter where we start 
    data = generateFractalImage(a = 0, b = 0) 
    #data2 = generateFractalImage(a = 1, b = 1) 

    # Plot fractal 
    fig, ax = plt.subplots(num = 1, nrows = 1, ncols = 1, figsize = (5, 5), dpi = 1200)
    ax.scatter(data[0], data[1], s = 1, c = 'DarkGreen', marker = '.') 
    #ax.scatter(data2[0], data2[1], s = 1, c = 'DarkRed', marker = '.') 
    #plt.axis('off')
    plt.tight_layout()
    fig.savefig("fern.png", dpi = 1200)
    plt.show() 
