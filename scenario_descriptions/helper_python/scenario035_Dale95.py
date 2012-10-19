
import mreorg
import sys

import numpy as np
import pylab as pl
import itertools


np.set_printoptions(precision=3, suppress=True)



#ca_state_vars = { "m": {"alpha":   [4.05,          0.0,  1.0 , -15.32, -13.57], 
#                          "beta1": [0.093 * 10.63, 0.093, -1,   10.63,  1.0], 
#                          "beta2": [1.28,          0,    1.0,    5.39, 12.11] } 
#                          }

MHMOD = 0.33544896 # (1.198032 * 0.28)
_alpha = lambda V_: 4.05 / (1.0+np.exp( -(V_-15.32)/13.57 ))
_beta1 = lambda V_: (1.28+MHMOD) / (1.0+np.exp( (V_+5.39)/12.11))
_beta2 = lambda V_: 0.093 * (V_+10.63) / (np.exp(V_+10.63) -1.0 )
_beta = lambda V_: _beta1(V_) if V_> -25.0 else _beta2(V_)

_inf = lambda V_: _alpha(V_)/(_alpha(V_)+_beta(V_))
_tau = lambda V_: 1.0/(_alpha(V_)+_beta(V_))

inf = np.frompyfunc(_inf,1,1)
tau = np.frompyfunc(_tau,1,1)
beta = np.frompyfunc(_beta,1,1)




if False:
    V = np.linspace(-80,60,num=200)
    betaV = beta(V)
    pl.plot(V, betaV , 'gx-')
    pl.show()








# vbar = (zVF)/(RT)
# where:
z = 2.         # -
F = 96485.3365  #(C/mol)
R = 8.3144621   #J K-1 mol-1
T = 300.         # 
# So if V is given in mV, then 
vbar = lambda V_: (V_ * 1.e-3) * (z * F) / (R * T)

if False:
    V = np.linspace(-80.0, 60., 100)
    pl.plot( V, vbar(V) )
    pl.show()
# which ranges from ~-6 to +6 for -80mV to 60mV


#print 'eta(-20)', vbar(-20.)
#print 'exp(-eta(-20))', np.exp(-1* vbar(-20.) )
#assert False

#
def ICA_in_mA_per_cm2_ungated(V_in_mV, PCA_in_cm_per_sec, CAI_in_nM, CAO_in_mM):
    PCA_in_m_per_sec = PCA_in_cm_per_sec * 1.e-2
    CAI_in_M = CAI_in_nM * (1.e-9)
    CAO_in_M = CAO_in_mM * (1.e-3)

    vb = vbar(V_in_mV)
    
    litres_per_metre3 = 1.e3

    ICA_in_A_per_m2 = litres_per_metre3 * PCA_in_m_per_sec * (z*F*vb) * ( CAI_in_M - CAO_in_M * np.exp(-vb) ) / (1. - np.exp(-vb) ) if np.fabs(V_in_mV) > 0.00001 else ( litres_per_metre3 *  PCA_in_m_per_sec * z * F * (CAI_in_M-CAO_in_M) ) 

    ICA_in_A_per_cm2 = ICA_in_A_per_m2 / (1.e4)
    ICA_in_mA_per_cm2 = ICA_in_A_per_cm2 * (1.e3)

    return ICA_in_mA_per_cm2


PCAs = [0.01, 0.03]
CAIs  = [100.,5000.]
CAOs  = [10.,20.]
VCMDs= [-80., -40., -20., 0., 20. ] 

for (VCMD_, PCA_, CAO_) in itertools.product( VCMDs, PCAs, CAOs ):
    CAI_ = 100.
    
    ica_mA_per_cm2_ungated =  ICA_in_mA_per_cm2_ungated( VCMD_, PCA_, CAI_, CAO_ )
    ica_mA_per_cm2 =  ica_mA_per_cm2_ungated * inf(VCMD_)**2
    
    lk = 0.03333333 * U.mS/U.cm2
    
    print VCMD_, '|',  PCA_, '|', CAI_, '|', CAO_, '|',ica_mA_per_cm2_ungated , '|', ica_mA_per_cm2, '|', ica_mA_per_cm2 * 20.*1e3
    #print 'Blah', ica_mA_per_cm2_ungated
    
    
#print ICA_in_mA_per_cm2( -30, 1, 100, 10 )
sys.exit(0)    


V = np.linspace(-80, 40, num=1000)

pl.figure()
for (PCA_, CAI_, CAO_) in itertools.product(PCAs, CAIs, CAOs):
    func = np.frompyfunc( lambda V:  ICA_in_mA_per_cm2_ungated( V, PCA_, CAI_, CAO_ ),1,1 )
    pl.plot(V, func(V), label='PCA:%f, CAI:%f CAO:%f' % (PCA_,CAI_,CAO_ ) )
pl.legend(loc=4)




pl.figure()
func = np.frompyfunc( lambda V:  ICA_in_mA_per_cm2_ungated( V, 1.0, 100, 10 ),1,1 )
pl.plot(V, func(V)*inf(V)**2, label='PCA:%f, CAI:%f CAO:%f' % (PCA_,CAI_,CAO_ ) )
pl.legend()
pl.show()





print 'm_inf:'
V = np.array( (-80., -60, -40, -20, 0, 20, 40) )
m_inf = inf(V)
print np.vstack( (V,m_inf)).T




# Plot Inf/Tau
if False: # or True:
    V = np.array( (-80., -60,-40,-20, 0, 20, 40) )
    V = np.linspace( -80, 60, num=100)
    infV = inf(V)
    tauV = tau(V)
    pl.figure()
    pl.plot(V,infV, 'xb-', label='m_inf')
    pl.plot(V,infV**2., 'xg-', label='m_inf**2')
    pl.figure()
    pl.plot(V,tauV, 'x-')
    pl.show()





pl.show()


