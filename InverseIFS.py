#!/usr/bin/env python3 

# InverseIFS.py 
# Author: Alexander Lind 
# Using points of a fractal as learning data for gradient descent 
# to get the affine transformations for the fractal (the inverse problem)

import sys
import numpy as np
from matplotlib import pyplot as plt

import IteratedFunctionSystem as IFS
from FractalImage import generateFractalImage 

def drawProgressBar(percent, barLength = 20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLength):
        if i < int(barLength * percent):
            progress += "="
        else: 
            progress += " "
    sys.stdout.write("[%s] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

if __name__ == "__main__": 

    # Generate target data 
    dataSize = 1000
    print('Generating target data with size = ' + str(dataSize))
    target = generateFractalImage(dataSize) 
    target_x = target[0]
    target_y = target[1]

    # Initialise random IFS 
    numAffineTransformations = 4
    print('Initialising random IFS with ' + str(numAffineTransformations) + ' affine transformations')
    ifs = IFS.IteratedFunctionSystem() 
    ifs.addRandomTransformations(numAffineTransformations) 

    # Train IFS on target data 

    print('\nTraining IFS on target data...')

    # Number of training steps 
    steps = 2 

    # Threshold distance for outliers 
    threshold = 1000000000000.0 

    drawProgressBar(0.0)

    for n in range(steps): 

        # Start at a random position in space (note that the IFS is a fixed attractor) 
        x = [np.random.rand()]
        y = [np.random.rand()]

        # Container with the affine functions used 
        affineFunctionsList = [] 

        # Container with the minimum differences 
        diffs = [] 

        # For each data point in the new generated data 
        for i in range(dataSize): 

            # Generate new data point cloud for our IFS 
            nf, xp, yp = ifs.evaluate(x[i - 1], y[i - 1])
            x.append(xp)
            y.append(yp)

            minDistance = 1.0e+308 # Overkill perhaps? 
            minDiff = (0.0, 0.0)  

            # For each data point in the target data 
            # in order to find distance to nearest neighbour 
            for j in range(dataSize): 

                # Calculate the difference necessary for the derivative of the error 
                diff = (xp - target_x[j], yp - target_y[j]) 

                # Calculate the distance (error/loss) 
                distance = 0.5 * diff[0]**2 + 0.5 * diff[1]**2 

                if (distance < minDistance): 
                    minDistance = distance
                    minDiff = diff 

            affineFunctionsList.append(nf)
            diffs.append(minDiff) 

        # Update IFS Parameters 
        for i in range(dataSize): 
            ifs.updateParameters(affineFunctionsList[i], (x[i], y[i]), diffs[i], 0.001) 

        drawProgressBar((n + 1) / steps)

    print('\n') # For clean output 

    #print('Loss = ' + str(minDistanceSum))

    # Print resulting IFS 
    print('Result for IFS: \n')
    ifs.output() 

    print('Plotting results...')

    # Generate final results for our IFS 
    x = [0]
    y = [0]
    for i in range(1, dataSize): 
        xy = ifs.evaluate(x[i - 1], y[i - 1])
        x.append(xy[0]) 
        y.append(xy[1]) 

    # Plot results 
    fig, ax = plt.subplots(num = 1, nrows = 1, ncols = 2, figsize = (10, 5), dpi = 100)
    ax[0].scatter(target_x, target_y, s = 1, c = 'Blue', marker = '.') 
    ax[1].scatter(x, y, s = 1, c = 'Purple', marker = '.') 
    ax[0].set_title('Target')
    ax[1].set_title('Result')
    plt.tight_layout()
    #fig.savefig("result.png", dpi = 1200)
    plt.show() 

    print('Done!')
