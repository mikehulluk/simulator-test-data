import numpy as np
import pylab

d = np.loadtxt('out1.dat')
pylab.plot( d[:,0], d[:,1])
pylab.show()
