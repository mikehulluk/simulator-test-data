
import quantities as pq
import numpy as np

T = 300. * pq.K
R = 8.3144 * pq.J / ( pq.mol * pq.K)
z=2.0
F = 96485.3365 * (pq.C / pq.mol)

pca = 1.0 * pq.cm / pq.s
V = -40. * pq.millivolt

Cai = 100. *pq.nano * (pq.mol/pq.liter)
Cao = 10. *pq.micro * (pq.mol/pq.liter)

eta = (z * V * F ) / ( R * T)
print eta
print eta.rescale(pq.dimensionless)
exp_eta = np.exp(-eta.rescale(pq.dimensionless))
print exp_eta

ica = pca * z * eta * F * (Cai - Cao * exp_eta)/(1-exp_eta)
print ica.rescale(  pq.milliamp / pq.centimeter**2)




