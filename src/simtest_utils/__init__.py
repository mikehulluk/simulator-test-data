
import sys
import os

# Find the root_directory
rootdir = os.path.join( os.path.dirname(__file__), '../../')
rootdir = os.path.abspath(rootdir)
# -----------------------



# Add glob2 and clint to the path
# --------------------------------
glob2dir = os.path.join(rootdir, 'src/glob2/src/')
sys.path.append(glob2dir)
clintdir = os.path.join(rootdir, 'src/clint/')
sys.path.append(clintdir)
# --------------------------------



import os
import re
import decimal
import sys
import collections
import itertools


import glob2
#import clint
import numpy as np
import pylab
import configobj


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'



scenario_path = os.path.join(rootdir, 'scenario_descriptions')
output_path = os.path.join(rootdir, 'output')


from testfunctionfunctorgenerator import TableTestFunctor


def check_all_scenarios():
    print 'Comparing Scenarios'
    for scenario_file in sorted( os.listdir(scenario_path) ):
        if not scenario_file.endswith('.txt'):
            continue
            
        # Only first for now:
        if not 'scenario001' in scenario_file:
        #if not 'scenario021' in scenario_file:
        #if not 'scenario075' in scenario_file:
            continue
        check_scenario( os.path.join( scenario_path, scenario_file ) )





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
    for rawline in data:

        # We can have commas in the input, so we need to split by commas:
        for line in itertools.product(*[ (l.split(",") if l is not None else [None]) for l in rawline ]):

            paramtuple = ParamTuple( **dict([ (var,decimal.Decimal(line[input_table_indices[var]])) for var in table_inputs ] ) )

            # Build functors to evaluate the output:
            for output in table_outputs:
                value = line[output_table_indices[output]]
                if value is None:
                    continue
                value = float(value)
                valiation = TableTestFunctor(test_expr=output, expected_value=value, eps=eps)
                validations[paramtuple].append(valiation)

    return validations




def check_scenario(scenario_file):

    print 'Checking Scenario from', scenario_file
    config = configobj.ConfigObj(scenario_file)
    print config['title']

    # Generate the file-list:
    scen_output_dir = os.path.join(output_path, config['scenario_short'] )
    file_list = [f for f in glob2.glob(scen_output_dir+'/**/*') if os.path.isfile(f) ]


    # Look at the configuration
    #  -- build a regex for the filenames:
    #  -- build a named-tuple object for holding parameters:
    parameters = config['Parameter Values']
    output_filename = config['Output Format']['base_filename']
    columns = config['Output Format']['columns']
    re_vars = re.compile(r"""<[A-Za-z0-9_]*>""")
    expected_variables = re_vars.findall(output_filename)
    expected_variables = [ exp_var[1:-1] for exp_var in expected_variables]
    expected_filename_regex = re_vars.sub(r'(-?[0-9]*(?:\.[0-9]*)?)', output_filename)
    expected_filename_regex = '(?:.*/)?' + expected_filename_regex + '(.*)'

    print ' -- Parameters', parameters.keys()
    print ' -- Output Columns', columns
    print ' -- Output Filename:', "'%s'" % output_filename
    print '    * Expected Variables:', expected_variables
    print '    * Expected Filename Regex:', expected_filename_regex


    # Compile the filename regular expression, and create a named-tuple for
    # storing the parameter values:
    filename_regex = re.compile(expected_filename_regex  )
    ParamTuple = collections.namedtuple('ParamTuple', expected_variables )

    # Sanity Check: Variables in filename tie up with those in [Parameter Values] block:
    if set(expected_variables) != set(parameters.keys()):
        print set(expected_variables), set(parameters.keys() )
        assert False, 'Parameters do not match filename template'


    # Look at all output files and find all the implementations:
    # Categorise all output files into a map impl_param_filename_dict[impl][param][filename]
    print ' -- Searching Files:'
    print '   * %d Files Found' % len(file_list)
    impl_param_filename_dict = collections.defaultdict(dict)
    unexpected_files = []
    for filename in file_list:
        m = filename_regex.match(filename)
        if m:
            params, impl = m.groups()[:-1], m.groups()[-1]
            params = ParamTuple(**dict([(var_name, decimal.Decimal(param)) for (var_name, param) in zip(expected_variables, params)]))
            assert not params in impl_param_filename_dict[impl], 'Duplicate Parameters foudn!'
            impl_param_filename_dict[impl][params] = filename
        else:
            unexpected_files.append(filename)

    for impl,params in impl_param_filename_dict.iteritems():
        print '    * %d files found for %s'% (len(params), impl)
    print '    * %d unexpected files found' % len(unexpected_files)

    for unexpected_file in unexpected_files:
        print bcolors.FAIL, '      > ', unexpected_file, bcolors.ENDC


    # Build an dictionary mapping {params -> {impl: filename, impl:filename} }
    # param_impl_filename[param][impl] -> filename
    all_params = set( itertools.chain(*[ v.keys() for v in impl_param_filename_dict.values()] ) )
    print '  * Parameter Sets Found', len(all_params)
    param_impl_filename_dict = {}
    for param in all_params:
        impl_with_param = [impl for impl in impl_param_filename_dict.keys() if param in impl_param_filename_dict[impl] ]
        param_impl_filename_dict[param] = {}
        for impl in impl_with_param:
            param_impl_filename_dict[param][impl] = impl_param_filename_dict[impl][param]

    # Parameter Evaluators:
    validators = {}
    # Look at the expect-values table:
    if 'Check Values' in config:
        print ' -- Building Validation Table'
        eps = float( config['Check Values']['eps'] )

        expectation_tables = [ k for k in config['Check Values'] if k.startswith('expect') and not k.endswith('_eps') ]
        for tbl_name in expectation_tables:
            local_eps = float( config['Check Values'].get(tbl_name+'_eps') or eps )
            print '    * Loading expectations:', k
            table_str = config['Check Values'][tbl_name]
            vals = parse_table(table_str, ParamTuple, variables=expected_variables,eps=local_eps)

            # Merge into to dictionary:
            for k, v in vals.iteritems():
                validators[k] = validators.get(k,[]) + v



    # Do the output:
    # #########################
    # Plot Comparitive graphs:
    n_traces = len(columns) - 1
    for param, impl_data in param_impl_filename_dict.iteritems():
        f = pylab.figure()
        f.suptitle('For Parameters: %s' % str(param))
        axes = [f.add_subplot(n_traces,1,i+1) for i in range(n_traces) ]

        # Plot the data:
        for (impl,filename) in impl_data.iteritems():
            data=  np.loadtxt(filename)
            for i in range(n_traces):
                axes[i].plot( data[:,0], data[:,i+1], label='%s-%s'%(impl, columns[i+1]), linewidth=2, alpha=0.5,  )

        # Smarten up the axes:
        for i, ax in enumerate(axes):
            ax.set_xmargin(0.05)
            ax.set_ymargin(0.05)
            ax.legend()
            ax.set_ylabel( columns[i+1] )
            
    #pylab.show()

    if validators:
        for impl, param_filename_dict in impl_param_filename_dict.iteritems():
            print '   * Checking Implementation Values against tables: %s' %impl
            for parameter, _validators in validators.iteritems():
                if not parameter in param_filename_dict:
                    print bcolors.WARNING, '        * Missing Parameters:',parameter, bcolors.ENDC
                    continue

                print '       * Checking against parameters:',parameter
                # Load the data:
                data = np.loadtxt(param_filename_dict[parameter])
                for validator in _validators:
                    result, message = validator.check_data(data, colnames=columns)
                    print (bcolors.FAIL if not result else bcolors.OKGREEN),
                    print  '           - ', result, message, bcolors.ENDC



    pylab.show()




