import os

if not os.path.exists('output/'):
    assert False, 'Output dir missing'
os.system('rm -f output/*')


os.system('nrniv -c A=10000 -c LK_G=0.1 -c LK_EREV=-51 -c C=1.0 -c VS=-31 -c I=100 test1.hoc > /dev/null 2>&1')
