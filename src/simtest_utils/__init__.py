
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



class Locations(object):
    @classmethod
    def scenario_descriptions(cls):
        return os.path.join(rootdir, 'scenario_descriptions')
    @classmethod
    def output_root(cls):
        return os.path.join(rootdir, 'output')



from testfunctionfunctorgenerator import TableTestFunctor



def check_scenarios(**kwargs):
    import waf_util
    scen_filenames = waf_util.get_all_scenarios()

    results = []
    for tgt_scen in waf_util.get_target_scenarios():
        tgt_scen_fname = scen_filenames[tgt_scen]
        res = check_scenario( tgt_scen_fname, **kwargs )
        results.append(res)
    return results







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

    # The header can have an optional '(eps:XXX)' in it. So lets parse that out:
    header_eps = {}
    re_header_eps = re.compile(r"""\(eps=(?P<eps>[^)*]*)\)""")
    for i,h in enumerate(header):
        m = re_header_eps.search(h)
        if m:
            new_header = re_header_eps.sub('', h).strip()
            header[i] = new_header
            header_eps[new_header] = float( m.groupdict()['eps'] )
        else:
            header_eps[h] = eps
    header = [h.strip() for h in header]


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
                col_eps = header_eps[output]
                valiation = TableTestFunctor(test_expr=output, expected_value=value, eps=col_eps)
                validations[paramtuple].append(valiation)

    return validations




def check_scenario(scenario_file, create_mredoc=True):

    print 'Checking Scenario from', scenario_file
    config = configobj.ConfigObj(scenario_file)
    scenario_short = config['scenario_short']
    scenario_title = config['title']
    print scenario_short
    print scenario_title

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
            assert not params in impl_param_filename_dict[impl], 'Duplicate Parameters found!'
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
    figures = {}
    for param, impl_data in param_impl_filename_dict.iteritems():
        f = pylab.figure()
        f.suptitle('For Parameters: %s' % str(param))
        axes = [f.add_subplot(n_traces,1,i+1) for i in range(n_traces) ]


        
        # Plot the data:
        common_time = None
        data_limits = [ None]  * n_traces
        for (impl,filename) in impl_data.iteritems():
            data=  np.loadtxt(filename)
            
            if common_time is None:
                common_time = np.arange( data[0,0], data[-1,0],  0.1)
                #common_time = data[:,0]
                
            for i in range(n_traces):
                axes[i].plot( data[:,0], data[:,i+1], label='%s-%s'%(impl, columns[i+1]), linewidth=2, alpha=0.5,  )
    
                # Extract the min and maxes:
                data_in_common_time = np.interp(common_time, data[:,0], data[:,i+1])
                if data_limits[i] is None:
                    data_limits[i] = data_in_common_time, data_in_common_time
                else:
                    mn = np.minimum(data_in_common_time,data_limits[i][0])
                    mx = np.maximum(data_in_common_time,data_limits[i][1])
                    data_limits[i] = (mn,mx)



        # Plot the discrepancy:
        for i in range(n_traces):
            axes[i].fill_between(common_time, data_limits[i][0], data_limits[i][1], color='red', facecolor='red', alpha=0.6)

        # Smarten up the axes:
        for i, ax in enumerate(axes):
            ax.set_xmargin(0.05)
            ax.set_ymargin(0.05)
            ax.legend()
            ax.set_ylabel( columns[i+1] )

        # Save the figures:
        figures[param] = f

    table_results = {}
    missing_parameter_sets = []
    if validators:
        for impl, param_filename_dict in impl_param_filename_dict.iteritems():
            print '   * Checking Implementation Values against tables: %s' %impl
            for parameter, _validators in validators.iteritems():
                if not parameter in param_filename_dict:
                    print bcolors.WARNING, '        * Missing Parameters:',parameter, bcolors.ENDC
                    missing_parameter_sets.append( (impl, parameter) )
                    continue

                print '       * Checking against parameters:',parameter
                # Load the data:
                data = np.loadtxt(param_filename_dict[parameter])
                for validator in _validators:
                    result, message, calc_value = validator.check_data(data, colnames=columns)
                    print (bcolors.FAIL if not result else bcolors.OKGREEN),
                    print  '           - ', result, message, bcolors.ENDC

                    table_results[impl, parameter, validator.test_expr] = (result, message, calc_value, validator)

    #pylab.show()
    if not create_mredoc:
        return None


    print ' -- Producing mredoc output'
    import mredoc as mrd

    comparison_graphs = mrd.Section('Comparison Graphs',
        [ mrd.Section(str(param), mrd.Image(fig, auto_adjust=False)) for (param,fig) in sorted(figures.items()) ]
        )


    tbl_comp_sections = []
    for impl in  impl_param_filename_dict:
        s = build_mredoc_results_table(impl=impl, validators=validators, table_results=table_results, missing_parameter_sets=missing_parameter_sets, expected_variables=expected_variables)
        tbl_comp_sections.append(s)



    return mrd.SectionNewPage('Results of Scenario: %s' % scenario_title,
        mrd.VerbatimBlock(config['description'] ) ,
        mrd.Paragraph('Tesing against: %s' % ', '.join(impl_param_filename_dict) ),
        mrd.TableOfContents(),
        mrd.Section('Table Comparison', *tbl_comp_sections),
        comparison_graphs,
    )


    pylab.show()




def build_mredoc_results_table(impl, validators, table_results, missing_parameter_sets, expected_variables):
        import mredoc as mrd

        output_cols = set(itertools.chain(*[[v.test_expr for v in V] for V in validators.values()]))
        input_cols = sorted(expected_variables)
        output_cols = sorted(output_cols)

        tbl_res = []
        for param in sorted(validators.keys()):
            in_vals = [ str(getattr(param,c)) for c in input_cols]
            out_vals = []
            for output_col_index, output_col in enumerate(output_cols):
                key = (impl, param, output_col)
                if key in table_results:
                    R = table_results[key]

                    if table_results[key][0]:
                        res = 'OK %f (%s [eps:%s])' % ( R[2],R[3].expected_value, R[3].eps  )
                    else:
                        res = '***ERROR %f (%s [eps:%s]) ***' % ( R[2],R[3].expected_value, R[3].eps  )
                else:

                    if (impl,param) in missing_parameter_sets:
                        res = ' *** MISSING! *** '
                    else:
                        res = '-'
                out_vals.append(str(res) )

            tbl_res.append(in_vals+out_vals)


        headers = input_cols + output_cols

        res_tbl = mrd.VerticalColTable( headers, tbl_res)

        impl_sect = mrd.Section('Table of Results for: %s' % impl,
                res_tbl,
                )
        return impl_sect
