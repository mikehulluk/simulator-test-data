

import quantities as pq
import numpy as np
import pylab as pl
from mhlibs.quantities_plot import QuantitiesFigure

from morphforge import units as mfu


T = 300. * pq.K
R = 8.3144 * pq.J / ( pq.mol * pq.K)
z=2.0
F = 96485.3365 * (pq.C / pq.mol)

pca = 1.0 * pq.cm / pq.s
pca = pca / 100.

SA = 1000. * pq.um **2

V = np.linspace(-80, 60, num=100) *  pq.millivolt

Cai = 100. *pq.nano * (pq.mol/pq.liter)
Cao = 10. *pq.micro * (pq.mol/pq.liter)

eta = (z * V * F ) / ( R * T)
print eta
print eta.rescale(pq.dimensionless)
exp_eta = np.exp(-eta.rescale(pq.dimensionless))
print exp_eta




ungated_ica = pca * z * eta * F * (Cai - Cao * exp_eta)/(1-exp_eta)
ungated_ica = ungated_ica.rescale(  pq.milliamp / pq.centimeter**2)

ungated_ica = ungated_ica* SA

g = np.gradient(ungated_ica) / np.gradient(V) 
g = g.rescale( mfu.pS)
R = (1/g).rescale(mfu.MOhm)
#print g


erev = V - (ungated_ica /g)
print erev


f = QuantitiesFigure()
ax1 = f.add_subplot(3,1,1)
ax2 = f.add_subplot(3,1,2)
ax3 = f.add_subplot(3,1,3)
ax1.plot( V, ungated_ica)
ax2.plot( V, R)
ax3.plot( V, erev)

pl.show()
