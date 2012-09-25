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
        # Only first for now:
        if not 'scenario001' in scenario_file:
            continue
        check_scenario( os.path.join( scenario_path, scenario_file ) )



def parse_header_output_functor(expr, value, eps):
    def check_func(data_matrix, colnames):
        return  "".join( ['Check that:', expr, 'is ', value, '(eps:',eps,')'])
    return check_func



def parse_table(table_str, ParamTuple, variables, eps):
    """ Returns a  map of {ParamTuple -> [ validating_func, validating_func ]}"""

    # Remove comments, and '|' at the beginning/end of lines:
    comment_regex = re.compile("#.*\n")
    table_end_regex = re.compile('(^\|)|(\|$)')
    table_sep_regex = re.compile('^[|-]*$')

    # Clean up the table:
    table_str = comment_regex.sub('\n', table_str)
    table_lines = [ line.strip() for line in table_str.split("\n") ]
    table_lines = [ table_end_regex.sub('',line) for line in table_lines]
    table_lines = [ table_sep_regex.sub('',line) for line in table_lines]
    table_lines = [ line for line in table_lines if line ]
    #print '\n'.join(table_lines)

    table_lines = [line.split("|") for line in table_lines]
    table_lines = [[tok.strip() for tok in line] for line in table_lines]
    table_lines = [[(tok if tok !='?' else None) for tok in line] for line in table_lines]
    header, data= table_lines[0], table_lines[1:]

    # Look at what in the header is a input, and what is an output:
    table_inputs = set(variables) & set(header)
    table_outputs = set(header).difference( set(variables))
    assert len(table_inputs) == len(variables), 'Table does not cover all variables!'

    input_table_indices = dict([(var, header.index(var)) for var in table_inputs])
    output_table_indices = dict([(var, header.index(var)) for var in table_outputs])

    # Construct the validations:
    validations = collections.defaultdict(list)
    for line in data:
        # Build a parameter tuple of the input:
        paramtuple = ParamTuple( **dict([ (var,decimal.Decimal(line[input_table_indices[var]])) for var in table_inputs ] ) )

        # Build functors to evaluate the output:
        for output in table_outputs:
            value = line[output_table_indices[output]]
            if value is None:
                continue
            valiation = parse_header_output_functor(expr=output, value=value, eps=eps)
            validations[paramtuple].append(valiation)

    return validations




def check_scenario(scenario_file):

    print 'Checking Scenario from', scenario_file
    config = configobj.ConfigObj(scenario_file)
    print config['title']

    scen_output_dir = os.path.join(output_path, config['scenario_short'] )
    simulators = os.listdir(scen_output_dir)
    print ' -- Simulators found', simulators


    # Look at the configuration
    #  -- build a regex for the filenames:
    #  -- build a named-tuple object for holding parameters:
    parameters = config['Parameter Values']
    output_filename = config['Output Format']['base_filename']
    columns = config['Output Format']['columns']
    print ' -- Parameters', parameters.keys()
    print ' -- Output Columns', columns
    print ' -- Output Filename:', "'%s'" % output_filename
    re_vars = re.compile(r"""<[A-Za-z0-9_]*>""")
    expected_variables = re_vars.findall(output_filename)
    expected_variables = [ exp_var[1:-1] for exp_var in expected_variables]
    expected_filename_regex = re_vars.sub(r'(-?[0-9]*(?:\.[0-9]*)?)', output_filename) + '(.*)'
    print '    * Expected Variables:', expected_variables
    print '    * Expected Filename Regex:', expected_filename_regex
    filename_regex = re.compile(expected_filename_regex  )
    ParamTuple = collections.namedtuple('ParamTuple', expected_variables )

    # Sanity Check: Variables in filename tie up with those in [Parameter Values] block:
    if set(expected_variables) != set(parameters.keys()):
        print set(expected_variables), set(parameters.keys() )
        assert False, 'Parameters do not match filename template'





    # Map ParamTuple objects to a dictionary:{Impl:filename}
    # (Look for generated files on the hard-disk)
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


    # Iterate through our ParamTuple objects.
    # For each, plot all the different implementation
    print ' -- Inspecting Found Files for plotting:'
    for param, impls in params_to_files.iteritems():
        print '   * ', param
        n_traces = len(columns) -1
        f = pylab.figure()
        f.suptitle('For Parameters: %s' % str(param))
        axes = [ f.add_subplot(n_traces,1,i+1) for i in range(n_traces) ]

        # Plot the data:
        for (impl,filename) in impls.iteritems():
            print '       ->', impl
            data=  np.loadtxt(filename)
            for i in range(n_traces):
                axes[i].plot( data[:,0], data[:,i+1],linewidth=2, alpha=0.5, label='%s-%s'%(impl, columns[i+1]) )

        # Smarten up the axes:
        for i, ax in enumerate(axes):
            ax.set_xmargin(0.05)
            ax.set_ymargin(0.05)
            ax.legend()


    #for param in params_to_files:
    #    print '::Param::', param
    # Look at the expect-values table:
    if 'Check Values' in config:
        print ' -- Building Validation Table'
        eps = config['Check Values']['eps']
        table_str = config['Check Values']['expectations']
        validators = parse_table(table_str, ParamTuple, variables=expected_variables,eps=eps)

        print ' -- Evaluating Validation Table'
        for param, validators in validators.iteritems():
            if not param in params_to_files:
                print '     >> !! No Traces found to evalute against!', param
            else:
                print '     >> Evaluating ', param
                impls = params_to_files[param]
                for implname, filename in impls.iteritems():
                    print '        ** Vaidating traces in :', filename
                    data_matrix = np.loadtxt(filename)
                    for v in validators:
                        print '          * ', v(data_matrix, colnames=columns)
        print


        return





    pylab.show()



        #print files





