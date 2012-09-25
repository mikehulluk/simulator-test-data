import os
import configobj
import re
import decimal
import collections
import numpy as np
import pylab

rootdir = os.path.join( os.path.dirname(__file__), '../../')
rootdir = os.path.abspath(rootdir)

scenario_path = os.path.join(rootdir, 'scenario_descriptions')
output_path = os.path.join(rootdir, 'output')

def check_all_scenarios():
    print 'Comparing Scenarios'
    for scenario_file in sorted( os.listdir(scenario_path) ):
        check_scenario( os.path.join( scenario_path, scenario_file ) )


re1 = re.compile(r"""<[A-Za-z0-9_]*>""")

def check_scenario(scenario_file):
    # Only first for now:
    if scenario_file != '/home/michael/hw_to_come/simulator-test-data/scenario_descriptions/scenario001_passive_singlecompartment.txt':
        return

    print 'Checking Scenario from', scenario_file
    config = configobj.ConfigObj(scenario_file)
    print config['title']

    scen_output_dir = os.path.join(output_path, config['scenario_short'] )
    simulators = os.listdir(scen_output_dir)
    print ' -- Simulators found', simulators

    parameters = config['Parameter Values']
    output_filename = config['Output Format']['base_filename']
    columns = config['Output Format']['columns']
    print ' -- Parameters', parameters.keys()
    print ' -- Output Columns', columns
    print ' -- Output Filename:', "'%s'" % output_filename

    #expected_filesnames = 
    expected_variables = re1.findall(output_filename)
    expected_variables = [ exp_var[1:-1] for exp_var in expected_variables]
    expected_filename_regex = re1.sub(r'(-?[0-9]*(?:\.[0-9]*)?)', output_filename) + '(.*)'
    print '    * Expected Variables:', expected_variables
    print '    * Expected Filename Regex:', expected_filename_regex
    filename_regex = re.compile(expected_filename_regex  )
    ParamTuple = collections.namedtuple('ParamTuple', expected_variables )
    

    # Sanity Check: Variables in filename tie up with those in [Parameter Values] block:
    if set(expected_variables) != set(parameters.keys()):
        print set(expected_variables), set(parameters.keys() )
        assert False, 'Parameters do not match filename template'

    # Map ParamTuple objects to a dictionary:{Impl:filename}
    params_to_files = {}
    print ' -- Searching for files:'
    for sim in simulators:
        print '   * Loading Data for:', sim
        sim_output_dir = os.path.join(scen_output_dir, sim)
        files = os.listdir(sim_output_dir)

        for filename in files:
            m = filename_regex.match(filename)
            if not m:
                print '      -> ERROR: Unable to parse:', filename
            else:
                print '      -> PARSED:', filename
                params, impl = m.groups()[:-1], m.groups()[-1]
                params = ParamTuple(**dict([(var_name, decimal.Decimal(param)) for (var_name, param) in zip(expected_variables, params)]))

                if not params in params_to_files:
                    params_to_files[params] = {}
                params_to_files[params][impl] = os.path.join(sim_output_dir,filename)

    print ' -- Inspecting Found Files:'
    for param, impls in params_to_files.iteritems():
        print '   * ', param
        n_traces = len(columns) -1
        f = pylab.figure()
        axes = [ f.add_subplot(n_traces,1,i+1) for i in range(n_traces) ]
        for ax in axes:
            ax.set_xmargin(0.05)
            ax.set_ymargin(0.05)

        for (impl,filename) in impls.iteritems():
            print '       ->', impl
            data=  np.loadtxt(filename)
            for i in range(n_traces):
                axes[i].plot( data[:,0], data[:,i+1],linewidth=2, alpha=0.5 )
            

    pylab.show()



        #print files





