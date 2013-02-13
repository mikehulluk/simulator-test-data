
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
import gc
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

import mredoc as mrd


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

    # Run the individual comparisons, and produce a section for
    # each:
    all_summary_results = []
    results_sections = []
    for tgt_scen in waf_util.get_target_scenarios():
        tgt_scen_fname = scen_filenames[tgt_scen]
        results_section, summary_results = check_scenario( tgt_scen_fname, **kwargs )
        results_sections.append(results_section)
        all_summary_results.extend(summary_results)
    sim_details =  mrd.Section('Simulation Details', *results_sections)


    # Produce an overall results table
    print 'Summary Results'
    for sum_res in all_summary_results:
        print sum_res


    sims = sorted( set( [ res.sim_name for res in all_summary_results] ) )
    scens = sorted( set( [ res.scen_name for res in all_summary_results]) )
    sim_scen_lut = dict([ ((res.sim_name, res.scen_name),res) for res in all_summary_results])

    def table_entry(scen,sim):
        key = (sim, scen)
        res = sim_scen_lut.get(key , None)
        #assert res is not None
        if res is None:
            return ":warning:[None Found]"
        prefix = ':success:' if not res.nmissing and not res.nfails else ''
        if res.nmissing:
            prefix = ':warning:'
        if res.nfails:
            prefix = ':err:'
        return prefix + res.summary_str()
        return str(res)

    header = [''] + sims
    res = [ ]
    for scen in scens:
        res.append( [scen] + [ table_entry(scen,sim) for sim in sims ] )

    overview_table = mrd.VerticalColTable( header, res)
    sim_overview = mrd.Section('Results Overview', overview_table )
    return (sim_details,sim_overview)






import hashlib


def cached_loadtxt(filename):
    assert os.path.exists(filename)

    h = hashlib.new('md5')
    with open(filename,'r') as f:
        h.update(f.read())
    cache_filename = filename + '_%s.npy' %h.hexdigest()

    if not os.path.exists(cache_filename):
        data = np.loadtxt(filename)
        np.save(cache_filename, data)
        return data
    else:
        return np.load(cache_filename)



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

    # The header can have an optional '(eps=XXX)' in it. So lets parse that out:
    header_eps = {}
    re_header_eps = re.compile(r"""\(eps=(?P<eps>[^)*]*)\)""")
    for i,h in enumerate(header):
        m = re_header_eps.search(h)
        if m:
            new_header = re_header_eps.sub('', h).strip()
            header[i] = new_header
            header_eps[new_header] =  m.groupdict()['eps']
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



class SimResults(object):
    def __init__(self, sim_name, scen_name):
        self.sim_name = sim_name
        self.scen_name = scen_name
        self.results = []
        self.missing_parameters = []
        self.found_parameters = []

    def add_table_result(self, result, eps_err, eps_err_pc):
        self.results.append( (result,eps_err, eps_err_pc) )

    def record_missing_parameters(self, parameter):
        self.missing_parameters.append(parameter)

    def record_found_parameters(self, parameter):
        self.found_parameters.append(parameter)

    def summary_str(self):
        return '%d Sucesses, %d Fails, %d Missing (Worst-Eps: %s %s%%)' % (self.nsuccesses, self.nfails, self.nmissing, self.worst_eps, self.worst_eps_pc)

    @property
    def nsuccesses(self):
        return len([s for s in self.results if s[0] ])
    @property
    def nmissing(self):
        return len(self.missing_parameters)
    @property
    def nfails(self):
        return len([s for s in self.results if not s[0] ])
    @property
    def worst_eps(self):
        return str(max( s[1] for s in self.results))[:5] if self.results else '='
    @property
    def worst_eps_pc(self):
        return str(max( s[2] for s in self.results))[:5] if self.results else '='

def check_scenario(scenario_file, create_mredoc=True):

    print 'Checking Scenario from', scenario_file
    config = configobj.ConfigObj(scenario_file)
    scenario_short = config['scenario_short']
    scenario_title = config['title']
    print scenario_short
    print scenario_title

    # Generate the file-list:
    scen_output_dir = os.path.join(output_path, config['scenario_short'] )
    file_list = [f for f in glob2.glob(scen_output_dir+'/**/*') if os.path.isfile(f) and not f.endswith('.npy') ]



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
    ParamTuple = collections.namedtuple('ParamTuple', sorted(expected_variables) )

    # Sanity Check: Variables in filename tie up with those in [Parameter Values] block:
    if set(expected_variables) != set(parameters.keys()):
        print set(expected_variables), set(parameters.keys() )
        assert False, 'Parameters do not match filename template'


    # Make a lsit of all the parameters we might expect to see:
    expected_parameters = set([])
    pnames = sorted(parameters.keys())
    pvals = [parameters[pname] for pname in pnames]
    for val in itertools.product(*pvals):
        expected_parameters.add( ParamTuple( **dict(zip(pnames,[decimal.Decimal(v) for v in val]))))
   

    #print 'Expected Parameters'
    #for a in sorted(expected_parameters):
    #    print a
    #print len(expected_parameters)





    # Look at all output files and find all the implementations:
    # Categorise all output files into a map impl_param_filename_dict[impl][param][filename]
    print ' -- Searching Files:'
    print '   * %d Files Found' % len(file_list)
    impl_param_filename_dict = collections.defaultdict(dict)
    unexpected_files = []
    unexpected_params = []
    for filename in file_list:
        m = filename_regex.match(filename)
        if m:
            params, impl = m.groups()[:-1], m.groups()[-1]
            params = ParamTuple(**dict([(var_name, decimal.Decimal(param)) for (var_name, param) in zip(expected_variables, params)]))
            assert not params in impl_param_filename_dict[impl], 'Duplicate Parameters found!'
            impl_param_filename_dict[impl][params] = filename
            if not params in expected_parameters:
                #print params
                #print params in expected_parameters
                #assert False
                unexpected_params.append((params,filename))
        else:
            unexpected_files.append(filename)

    for impl,params in impl_param_filename_dict.iteritems():
        print '    * %d files found for %s'% (len(params), impl)
    print '    * %d unexpected files found' % len(unexpected_files)

    for unexpected_file in unexpected_files:
        print bcolors.FAIL, '      > ', unexpected_file, bcolors.ENDC


    # Look at the implementations, and map them to known simulators:
    import waf_util
    known_simulators = waf_util.get_all_simulators()
    impl_to_sim_map = {}
    for impl_name in impl_param_filename_dict.keys():
        possible_sims = [ sim for sim in known_simulators if impl_name.startswith(sim)]
        assert len(possible_sims) == 1, "'%s' is not a suffix of the known simulators: %s " % (impl_name, ','.join(known_simulators))
        impl_to_sim_map[impl_name] = possible_sims[0]
    # Create a results object to hold the results for each simulator:
    simulator_results = dict( [(sim, SimResults(sim_name=sim,scen_name=scenario_short)) for sim in set(impl_to_sim_map.values())] )


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
        eps = config['Check Values']['eps']

        expectation_tables = [ k for k in config['Check Values'] if k.startswith('expect') and not k.endswith('_eps') ]
        for tbl_name in expectation_tables:
            local_eps = config['Check Values'].get(tbl_name+'_eps') or eps
            print '    * Loading expectations:', tbl_name
            table_str = config['Check Values'][tbl_name]
            vals = parse_table(table_str, ParamTuple, variables=expected_variables,eps=local_eps)

            # Merge into to dictionary:
            for k, v in vals.iteritems():
                validators[k] = validators.get(k,[]) + v


    from scipy.interpolate import interp1d
    # Load the files, and downsample them
    nfiles = sum( [ len(v) for v in impl_param_filename_dict.values() ] )
    print '  * Preloading and downsampling data (%d files) ' % nfiles
    stop = float( config['Sampling']['stop'] )
    dt = float( config['Sampling']['dt'] )
    common_time = np.arange( 0.0, stop, dt)
    preloaded_data = {}
    for impl, paramfilenamedict in impl_param_filename_dict.items():
        for (param, filename) in paramfilenamedict.items():
            print '.',
            sys.stdout.flush()
            data =  cached_loadtxt(filename)
            assert data.shape[1] == len(columns)
            original_time = data[:,0]
            data_downsampler = interp1d( original_time, data.T, kind='linear', copy=False) #, bounds_error=False )
            data_downsampled = data_downsampler(common_time).T
            preloaded_data[filename] = data_downsampled
            del data
    print
    print '    >> Invoking garbage collector'
    gc.collect()
    print '   >> Finished preloading'



    TraceCompRes = collections.namedtuple('TraceComparisonResult', ['max_ptp','mean_ptp' ] )
    # Calcuate the distances between the traces:
    trace_comp_res = {}
    for param, impl_data in param_impl_filename_dict.iteritems():
        trace_comp_res[param] = {}
        for i,colname in enumerate(columns):
            if i==0:
                continue

            # Load ll the data for a paramter for a column
            impls = sorted( impl_data.keys() )
            impl_cols = [preloaded_data[impl_data[impl]][:,i] for impl in impls]
            d = np.vstack( impl_cols)
            assert d.shape[0] == len(impls)
            assert d.shape[1] == common_time.shape[0]

            # cacluate the ptp difference at each time point:
            ptp = np.ptp(d, axis=0)
            assert ptp.shape[0] == common_time.shape[0]

            
            res = TraceCompRes( 
                    max_ptp= np.max(ptp),
                    mean_ptp= np.mean(ptp) )
            #largest_diff = np.max(ptp)
            #mean_diff = np.mean(ptp)
            trace_comp_res[param][colname] =  res
            #print largest_diff


            #print d.shape

            #print impl_cols
            #assert False

            #for impl,filename in impl_data.items():
                
            pass
        pass



    # Do the output:
    # #########################
    # Plot Comparitive graphs:
    print '  * Producing Comparison Graphs'
    n_traces = len(columns) - 1
    figures = {}
    for param, impl_data in param_impl_filename_dict.iteritems():
        f = pylab.figure()
        f.suptitle('For Parameters: %s' % str(param))
        axes = [f.add_subplot(n_traces,1,i+1) for i in range(n_traces) ]

        # Plot the data:
        data_limits = [ None]  * n_traces
        for (impl,filename) in impl_data.iteritems():
            data_all = preloaded_data[filename]

            for i in range(n_traces):
                d =  data_all[:,i+1]
                axes[i].plot( data_all[:,0], d, label='%s-%s'%(impl, columns[i+1]), linewidth=2, alpha=0.5,  )

                # Calculate the maxiumu ranges of the data:
                if data_limits[i] is None:
                    data_limits[i] = d, d
                else:
                    mn = np.minimum(d,data_limits[i][0])
                    mx = np.maximum(d,data_limits[i][1])
                    data_limits[i] = (mn,mx)



        # Plot the discrepancy:
        for i in range(n_traces):
            axes[i].fill_between(common_time, data_limits[i][0], data_limits[i][1], color='red', facecolor='red', alpha=0.6, label='Max distance between traces')

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
            for parameter, _validators in sorted(validators.iteritems()):

                # Parameter Missing:
                if not parameter in param_filename_dict:
                    print bcolors.WARNING, '        * Missing Parameters:',parameter, bcolors.ENDC
                    missing_parameter_sets.append( (impl, parameter) )
                    simulator_results[impl_to_sim_map[impl]].record_missing_parameters(parameter)
                    continue

                # Parameter Found:
                else:
                    simulator_results[impl_to_sim_map[impl]].record_found_parameters(parameter)

                    print '       * Checking against parameters:',parameter
                    data = preloaded_data[param_filename_dict[parameter] ]
                    for validator in _validators:
                        result, message, calc_value = validator.check_data(data, colnames=columns)
                        print ({False:bcolors.FAIL,True:bcolors.OKGREEN}[result]),
                        print  '           - ', result, message, bcolors.ENDC

                        table_results[impl, parameter, validator.test_expr] = (result, message, calc_value, validator)
                        eps_err = np.fabs( validator.expected_value - calc_value)
                        eps_err_pc = np.fabs((eps_err/ calc_value)) * 100.
                        simulator_results[impl_to_sim_map[impl]].add_table_result(result, eps_err = eps_err, eps_err_pc = eps_err_pc )


    summary_results = simulator_results.values()


    create_mredoc=True #and False
    results_section = None
    if create_mredoc:
        results_section = build_mredoc_ouput( trace_comp_res=trace_comp_res, unexpected_params=unexpected_params,config=config,figures= figures, validators= validators, table_results= table_results, missing_parameter_sets= missing_parameter_sets, expected_variables= expected_variables, impl_param_filename_dict= impl_param_filename_dict)


    # Return the detailed results, and what we will need to produce the summary graphs:
    return results_section, summary_results


def build_mredoc_ouput(trace_comp_res,unexpected_params, config, figures, validators, table_results, missing_parameter_sets, expected_variables, impl_param_filename_dict):
    print ' -- Producing mredoc output'
    import mredoc as mrd


    def build_comp_section(param):
        fig = figures[param]
        trcs = sorted( trace_comp_res[param] )
        comp_tbl_header, comp_tbl_data= zip( *[ ('%s.max_ptp' %c, trace_comp_res[param][c].max_ptp) for c in trcs])
        return mrd.Section('Parameters: %s' % str(param),
            mrd.VerticalColTable(comp_tbl_header, [comp_tbl_data] ),
            mrd.Figure( mrd.Image(fig, auto_adjust=False), caption='Comparison'),
            )

    comparison_graphs = mrd.Section('Trace Comparisons', [build_comp_section(p) for p in figures.keys() ] )

    tbl_comp_sections = []
    for impl in  impl_param_filename_dict:
        s = build_mredoc_results_table(impl=impl, validators=validators, table_results=table_results, missing_parameter_sets=missing_parameter_sets, expected_variables=expected_variables)
        tbl_comp_sections.append(s)

    results_section = mrd.SectionNewPage('%s - %s' % ( config['scenario_short'].capitalize(), config['title'],),
        mrd.Section('Overview',
            mrd.Section('Description', mrd.VerbatimBlock(config['description'], caption='Description' ) ),
            mrd.Section('Implementations', mrd.Paragraph('Tesing against: %s' % ', '.join(impl_param_filename_dict) ) ),
            mrd.Section('Failures', mrd.Paragraph('TODO!')),
            mrd.Section('Unexpected Files', mrd.Paragraph('Files:' + ','.join([str(u[1]) for u in unexpected_params]) )),
            ),
        mrd.Section('Table Comparisons',*tbl_comp_sections ),
        comparison_graphs,
    )

    return results_section






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

                    prefix = ''
                    if table_results[key][0]:
                        prefix=':success:'
                        res = 'OK %f (%s [eps:%s])' % ( R[2],R[3].expected_value, R[3].eps  )
                    else:
                        prefix=':err:'
                        res = '***ERROR %f (%s [eps:%s]) ***' % ( R[2],R[3].expected_value, R[3].eps  )
                else:

                    prefix = ''
                    if (impl,param) in missing_parameter_sets:
                        prefix=':warning:'
                        res = ' *** MISSING! *** '
                    else:
                        res = '-'
                out_vals.append(prefix + str(res))

            tbl_res.append(in_vals+out_vals)


        headers = input_cols + output_cols

        res_tbl = mrd.VerticalColTable( headers, tbl_res,caption='Results for: %s' % impl,)
        return res_tbl

        #impl_sect = mrd.Section
        #        res_tbl,
        #        )
        #return impl_sect
