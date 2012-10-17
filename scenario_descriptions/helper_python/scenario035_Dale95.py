
import numpy as np
import pylab as pl
import itertools


np.set_printoptions(precision=3, suppress=True)



ca_state_vars = { "m": {"alpha":   [4.05,          0.0,  1.0 , -15.32, -13.57], 
                          "beta1": [0.093 * 10.63, 0.093, -1,   10.63,  1.0], 
                          "beta2": [1.28,          0,    1.0,    5.39, 12.11] } 
                          }

_alpha = lambda V_: 4.05 / (1.0+np.exp( -(V_-15.32)/13.57 ))
_beta1 = lambda V_: 1.28 / (1.0+np.exp( (V_+5.39)/12.11)) 
_beta2 = lambda V_: 0.093 * (V_+10.63) / (np.exp(V_+10.63) -1.0 )
_beta = lambda V_: _beta1(V_) if V_> -25.0 else _beta2(V_)

_inf = lambda V_: _alpha(V_)/(_alpha(V_)+_beta(V_))
_tau = lambda V_: 1.0/(_alpha(V_)+_beta(V_))

inf = np.frompyfunc(_inf,1,1)
tau = np.frompyfunc(_tau,1,1)






# vbar = (zVF)/(RT)
# where:
z = 2           # -
F = 96485.3365  #(C/mol)
R = 8.3144621   #J K-1 mol-1
T = 300         # 
# So if V is given in mV, then 
vbar = lambda V_: z*(V_ * 1e-3) * F / (R * T)

if False:
    V = np.linspace(-80, 60, 100)
    pl.plot( V, vbar(V) )
    pl.show()
# which ranges from ~-6 to +6 for -80mV to 60mV



#
def ICA_in_mA_per_cm2(V_in_mV, PCA_in_cm_per_sec, CAI_in_nM, CAO_in_mM):
    PCA_in_m_per_sec = PCA_in_cm_per_sec * 1e-2
    CAI_in_M = CAI_in_nM * (1e-9)
    CAO_in_M = CAO_in_mM * (1e-3)

    vb = vbar(V_in_mV)

    ICA_in_A_per_m2 = PCA_in_m_per_sec * (z*F*vb) * ( CAI_in_M - CAO_in_M * np.exp(-vb) ) / (1. - np.exp(-vb) )
    ICA_in_A_per_cm2 = ICA_in_A_per_m2 / (1e4)
    ICA_in_mA_per_cm2 = ICA_in_A_per_cm2 * (1e3)

    return ICA_in_mA_per_cm2


PCAs = [1.,3.]
CAIs  = [100.,150.]
CAOs  = [10.,20.]
VCMDs= [-80., -40., 0., 20. ] 

for (VCMD_, PCA_, CAI_, CAO_) in itertools.product( VCMDs, PCAs, CAIs, CAOs ):
    
    ica_mA_per_cm2 =  ICA_in_mA_per_cm2( VCMD_, PCA_, CAI_, CAO_ )
    print VCMD_, PCA_, CAI_, CAO_, ica_mA_per_cm2
    
    
print ICA_in_mA_per_cm2( -30, 1, 100, 10 )
    









print 'm_inf:'
V = np.array( (-80., -60, -40, -20, 0, 20, 40) )
m_inf = inf(V)
print np.vstack( (V,m_inf)).T




# Plot Inf/Tau
if False or True:
    V = np.array( (-80., -60,-40,-20, 0, 20, 40) )
    V = np.linspace( -80, 60, num=100)
    infV = inf(V)
    tauV = tau(V)
    pl.figure()
    pl.plot(V,infV, 'xb-')
    pl.plot(V,infV**2., 'xg-')
    pl.figure()
    pl.plot(V,tauV, 'x-')





pl.show()


