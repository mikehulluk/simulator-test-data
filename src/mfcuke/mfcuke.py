

import glob
import configobj
import re
import itertools
import pylab
import os
import numpy as np
#import copy
import random


import quantities as pq
import morphforge.stdimports as mf
import morphforgecontrib.stdimports as mfc
from neurounits import NeuroUnitParser



import sys
module_dir = os.path.abspath( os.path.join( os.path.dirname(__file__), '../') )
if not module_dir in sys.path:
    sys.path.append(module_dir)
import simtest_utils


class UnhandledDescription(RuntimeError):
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return "No handlers found that recognise: '%s'" % self.expr

class ActionContext(object):
    def __init__(self, parameter_values=None):
        self.parameter_values = parameter_values or None
        self.obj_refs = {}
        self.records = {}

    def resolve_context_parameter_values(self, dct):
        for k,v in dct.copy().iteritems():
            v_name = is_context_parameter(v)
            if not v_name:
                continue
            dct[k] = self.parameter_values[v_name]
        return dct

class ActionHandleParent(object):
    def __init__(self, src_regex_str, child_handlers=None):

        self.src_regex_str = src_regex_str
        self.src_regex = re.compile(src_regex_str)
        self.child_handlers = child_handlers or []

        for child in self.child_handlers:
            child.parent_handler = self

    def __str__(self):
        print 'Handler for: %s', self.src_regex_str

    def can_handle(self, expr):
        return self.src_regex.match(expr)

    def parse(self, expr):
        return self.src_regex.match(expr)




class ActionCodePath(object):
    def __init__(self, description_lines, handlers):
        self.activities = zip(description_lines, handlers)

    def execute(self, ctx):
        for (desc_line,  child_handler) in self.activities:
            m = child_handler.parent_handler.parse(desc_line)
            assert m
            child_handler.__call__(ctx=ctx, groupdict=m.groupdict(), expr=desc_line)

class ActionHandlerLibrary(object):
    def __init__(self):
        self.handlers =[]

    def register_handler(self, handler):
        self.handlers.append(handler)

    def find_handlers(self, expr):
        return [ h for h in self.handlers if h.can_handle(expr)]

    def get_parent_handler(self, expr):
        handlers = self.find_handlers(expr)
        if len(handlers) == 0:
            raise UnhandledDescription(expr)
        assert len(handlers) == 1
        return handlers[0]

    def build_code_paths(self, description_lines, mode='all'):


        # Build a list of all possible functors for each line:
        all_child_handlers = []
        for i, line in enumerate(description_lines):
            h = self.get_parent_handler(line)
            if not h.child_handlers:
                raise RuntimeError('No child-handler specified for: %s' % line)
            all_child_handlers.append( h.child_handlers )


        # Take the outer-product to get all the code-paths:
        code_paths = []

        if mode == 'all':
            for hs in itertools.product(*all_child_handlers):
                code_paths.append(ActionCodePath(description_lines, hs) )
            return code_paths

        if mode == 'reduced':
            # Ensure that each handler is used at least once, and leave the 
            # rest to randomisation.
            unused_all_child_handlers = [ sl[:] for sl in all_child_handlers]

            for ch in unused_all_child_handlers:
                random.shuffle(ch)

            # Have we covered all bases?
            while(True):

                if not [sublist for sublist in  unused_all_child_handlers if sublist]:
                    break

                handlers = []
                for lineindex, line in enumerate(description_lines):
                    handler = None
                    if unused_all_child_handlers[lineindex]:
                        handler = unused_all_child_handlers[lineindex].pop()
                    else:
                        handler = random.choice(all_child_handlers[lineindex])
                    handlers.append( handler)

                code_paths.append(ActionCodePath(description_lines, handlers))
            return code_paths








def parse_param_str(expr):
    try:
        res = {}
        for param_tok in re.split(r"""(?:and)|(?:,)""", expr):
            toks = param_tok.strip().split()
            p0,p1 = toks[0], " ".join(toks[1:])
            assert not p0 in res
            res[p0] = p1
        return res
    except ValueError:
        raise ValueError('Error parsing: %s' % expr)

def is_context_parameter(expr):
    if not expr:
        return
    m = re.match(r"""\<(?P<name>.*)\>""", expr)
    if not m:
        return None
    return m.groupdict()['name']

def resolve_location(ctx, locstr):
    if '.' in locstr:
        cellname, pos = locstr.split('.')
        return ctx.obj_refs[cellname].get_location(pos)
    else:
        return ctx.obj_refs[locstr].soma

def convert_string_to_quantity(s):
    if isinstance(s, basestring):
        return NeuroUnitParser.QuantitySimple(s).as_quantities_quantity()
    else:
        return s

def convert_all_string_to_quantities(dct, keys=None):
    for k,v in dct.items():
        if not keys or k in keys:
            dct[k] = convert_string_to_quantity(v)
    return dct



# =====================================
def sim_builder_wrapper( func ):
    def new_func(self, ctx, expr, groupdict, ):
        assert set(groupdict.keys()) == set(['duration'])
        convert_all_string_to_quantities(groupdict)
        return func(self, ctx, sim_duration=groupdict['duration'])
    return new_func

class sim_builder_a(object):
    @sim_builder_wrapper
    def __call__(self, ctx, sim_duration):
        ctx.env = mf.NEURONEnvironment()
        ctx.sim = ctx.env.Simulation(tstop=sim_duration, cvode=False)
class sim_builder_b(object):
    @sim_builder_wrapper
    def __call__(self, ctx, sim_duration):
        ctx.env = mf.NEURONEnvironment()
        ctx.sim = ctx.env.Simulation(tstop=sim_duration, cvode=True)
# ====================================


# =====================================
def sim_run_wrapper( func ):
    def new_func(self, ctx, expr, groupdict):
        assert not groupdict
        return func(self, ctx,)
    return new_func

class sim_run(object):
    @sim_run_wrapper
    def __call__(self, ctx):
        ctx.res = ctx.sim.run()
        print ctx.res.get_traces()
        for recname, recobj in ctx.records.items():
            print recname, ctx.res.get_trace(recobj)
# =====================================



# =====================================
def sim_build_single_compartment_wrapper( func ):
    def new_func(self, ctx, expr, groupdict, ):
        assert set(groupdict.keys()) == set(['name','params'])
        name = groupdict['name']
        # Break the parameter-string into component parts, and resolve prefined variables:
        params = parse_param_str(groupdict['params'])
        params = ctx.resolve_context_parameter_values(params)
        convert_all_string_to_quantities(params, keys=['initialvoltage', 'capacitance','area'])

        return func(self,
                ctx,
                name=name,
                initialvoltage= params.get('initialvoltage',None),
                capacitance = params.get('capacitance',None),
                area=params.get('area',None))
    return new_func

class sim_build_single_compartment(object):
    @sim_build_single_compartment_wrapper
    def __call__(self, ctx, name, initialvoltage, capacitance, area ):
        create_cell_kwargs = {'name':name, 'area':area}
        if initialvoltage is not None:
            create_cell_kwargs['initial_voltage'] = initialvoltage
        cell = ctx.sim.create_cell(**create_cell_kwargs)
        if capacitance is not None:
            ctx.sim.get_cell(cellname=name).set_passive(mf.PassiveProperty.SpecificCapacitance, capacitance )
        ctx.obj_refs[name] = cell
# =====================================



# =====================================
def sim_record_wrapper( func ):
    def new_func(self, ctx, expr, groupdict):
        assert set(groupdict.keys()) == set(['where','what','as'])
        where = groupdict['where']
        what = groupdict['what']
        as_ = groupdict['as']
        return func(self, ctx,where=where, what=what,as_=as_)
    return new_func



class sim_record(object):
    @sim_record_wrapper
    def __call__(self, ctx, where, what, as_):
        assert where in ctx.obj_refs
        assert not as_ in ctx.records
        LUT_what={
                'V': mf.StandardTags.Voltage,
                'Conductance': mf.StandardTags.Conductance,
                'Current': mf.StandardTags.Current,
                }
        what_trans = LUT_what[what]
        r = ctx.sim.record(ctx.obj_refs[where], what=what_trans)
        ctx.records[as_] = r
# =====================================



# =====================================
def sim_step_current_injection_wrapper( func ):
    def new_func(self, ctx, expr, groupdict):
        assert set(groupdict.keys()) == set(['amplitude','cell','location', 'from','until'])

        # Resolve the groupdict into units for 'amp','for' and 'until'
        groupdict = ctx.resolve_context_parameter_values(groupdict)
        convert_all_string_to_quantities(groupdict, keys=['amplitude','from','until'])

        return func(self, ctx,
            amplitude = groupdict['amplitude'],
            from_ = groupdict['from'],
            until = groupdict['until'],
            cell = groupdict['cell'],
            location = groupdict['location'] or 'soma'
            )
    return new_func

class sim_step_current_injection(object):
    @sim_step_current_injection_wrapper
    def __call__(self, ctx, amplitude, from_, until, cell, location):
        cell = ctx.obj_refs[cell]
        ctx.sim.create_currentclamp(cell_location=cell.get_location(location),
                                    amp=amplitude,
                                    delay=from_,
                                    dur=until-from_)
# =====================================


# =====================================
def sim_add_channel_wrapper( func ):
    def new_func(self, ctx, expr, groupdict):
        assert set(groupdict.keys()) == set(['channelname','cell','params'])
        # Break the parameter-string into component parts, and resolve prefined variables:
        params = parse_param_str(groupdict['params'])
        params = ctx.resolve_context_parameter_values(params)

        return func(self,
                ctx,
                channelname=groupdict['channelname'],
                cell_name=groupdict['cell'],
                **params)
    return new_func

class sim_add_channel(object):
    @sim_add_channel_wrapper
    def __call__(self, ctx, channelname, cell_name, **kwargs):

        # Create the channel:
        if channelname == 'Leak':
            assert set( kwargs.keys() ) == set(['reversalpotential','conductance'])
            convert_all_string_to_quantities(kwargs)
            chl = ctx.env.Channel(mfc.StdChlLeak, conductance=kwargs['conductance'], reversalpotential=kwargs['reversalpotential'] )
        else:
            assert False

        # Apply the channel:
        cell = ctx.obj_refs[cell_name]
        cell.apply_channel(chl)
# =====================================



# =====================================
def sim_create_gap_junction_wrapper( func ):
    def new_func(self, ctx, expr, groupdict):
        assert set(groupdict.keys()) == set(['name','resistance','loc1','loc2'])
        groupdict = ctx.resolve_context_parameter_values(groupdict)

        return func(self, ctx,
                name=groupdict['name'],
                resistance=groupdict['resistance'],
                loc1=groupdict['loc1'],
                loc2=groupdict['loc2'])
    return new_func

class sim_create_gap_junction(object):
    @sim_create_gap_junction_wrapper
    def __call__(self, ctx, name, resistance, loc1, loc2):

        ctx.sim.create_gapjunction(
                name=name,
                celllocation1 = resolve_location(ctx,loc1),
                celllocation2 = resolve_location(ctx,loc2),
                resistance = resistance
                )

        pass
# =====================================


# =====================================
def sim_create_synapse_wrapper( func ):
    def new_func(self, ctx, expr, groupdict):
        assert set(groupdict.keys()) == set(['synapsetype','location','params','times','name'])
        synapsetype = groupdict['synapsetype']
        location = groupdict['location']
        params = groupdict['params']
        times = groupdict['times']
        name = groupdict['name']
        return func(self, ctx, name=name, synapsetype=synapsetype,location=location,params=params,times=times)
    return new_func

class sim_create_synapse(object):
    @sim_create_synapse_wrapper
    def __call__(self, ctx,name, synapsetype,location,params,times):
        target_cell_loc  = resolve_location(ctx,location)

        synkwargs = parse_param_str(params)
        synkwargs = ctx.resolve_context_parameter_values(synkwargs)
        convert_all_string_to_quantities(synkwargs)

        times_ms = [float(f) for f in times.split(',')]

        # Create the synapse_tmplate:
        if synapsetype == 'SingleExponential':
            syn_kw_mapped = {
                'tau': synkwargs['closing-time'],
                'peak_conductance': synkwargs['conductance'],
                'e_rev': synkwargs['reversalpotential'],
                    }
            syn_tmpl = ctx.env.PostSynapticMechTemplate(mfc.PostSynapticMech_ExpSyn_Base, **syn_kw_mapped)
        elif synapsetype == 'DoubleExponential':
            syn_kw_mapped = {
                'tau_open': synkwargs['closing-time'],
                'tau_close': synkwargs['opening-time'],
                'peak_conductance': synkwargs['conductance'],
                'e_rev': synkwargs['reversalpotential'],
                    }
            syn_tmpl = ctx.env.PostSynapticMechTemplate(mfc.PostSynapticMech_Exp2Syn_Base, popening=1.0, **syn_kw_mapped)
        else:
            assert False


        syn = ctx.sim.create_synapse(
                trigger = ctx.env.SynapticTrigger(mf.SynapticTriggerAtTimes, time_list=times_ms * pq.ms ),
                postsynaptic_mech = syn_tmpl.instantiate(cell_location=target_cell_loc)
                 )
        ctx.obj_refs[name] = syn
# =====================================






handler_lib = ActionHandlerLibrary()

handler_lib.register_handler( ActionHandleParent(
    src_regex_str = r"""In a simulation lasting (?P<duration>[\d.]*ms)""",
    child_handlers=[sim_builder_a(), sim_builder_b()]) )

handler_lib.register_handler( ActionHandleParent(
    src_regex_str = r"""Run the simulation""",
    child_handlers=[sim_run()]))

handler_lib.register_handler( ActionHandleParent(
    src_regex_str = r"""Create a single compartment neuron '(?P<name>\w+)' with (?P<params>.*)""",
    child_handlers=[sim_build_single_compartment()] ) )

handler_lib.register_handler( ActionHandleParent(
    src_regex_str = r"""Record (?P<where>[\w]*)\.(?P<what>[\w]+) as [$](?P<as>\w+)""",
    child_handlers=[sim_record()]))

handler_lib.register_handler( ActionHandleParent(
    src_regex_str = r"""Inject step-current of (?P<amplitude>.*) into (?P<cell>\w+)(\.(?P<location>\w+))? from t=(?P<from>.*) until t=(?P<until>.*)""",
    child_handlers=[sim_step_current_injection()]) )

handler_lib.register_handler( ActionHandleParent(
    src_regex_str = r"""Add (?P<channelname>\w+)* channels to (?P<cell>\w+)* with (?P<params>.*)""",
    child_handlers=[sim_add_channel()]) )

handler_lib.register_handler( ActionHandleParent(
    src_regex_str = r"""Create a gap junction '(?P<name>\w+)' with resistance (?P<resistance>.*) between (?P<loc1>[\w.]+) and (?P<loc2>[\w.]+)""",
    child_handlers=[sim_create_gap_junction()]  ) )

handler_lib.register_handler( ActionHandleParent(
    src_regex_str = r"""Create a (?P<synapsetype>.*) synapse '(?P<name>.*)' onto (?P<location>[\w.]*) with (?P<params>.*) driven with spike-times at \[(?P<times>.*)\]ms""",
    child_handlers=[sim_create_synapse()]) )








def run_scenario_filename(fname, code_path_mode='reduced', only_first_paramtuple=False, plot_results=False, short_run=False):
    
    if short_run is True:
        only_first_paramtuple = True
        
    
    print 'Reading from file', fname
    conf = configobj.ConfigObj(fname)


    units_dict = dict( [(k,NeuroUnitParser.Unit(v).as_quantities_unit() ) for (k,v) in conf['Units'].iteritems() ] )
    param_syms = conf['Parameter Values'].keys()
    param_vals = [ conf['Parameter Values'][sym] for sym in param_syms]


    #description_lines = [ l.strip() for l in conf['description'].split('\n')]    
    #description_lines = [ l for l in description_lines if l]
    
    description_lines = []
    for line in conf['description'].split('\n'):
        
        # Skip blank lines:
        if not line.strip():
            continue
            
        # If it starts with whitespace, then add it to the previous line
        if re.match(r"""^[\s]+""", line):
            description_lines[-1] = description_lines[-1] + ' ' + line.strip()
            continue
        
        # Otherwise, just add it
        description_lines.append(line)
        
    description_lines = [ l.strip() for l in description_lines]
    description_lines = [ re.sub(r"\s+", " ", l) for l in description_lines]
    
    
    

    code_paths = handler_lib.build_code_paths(description_lines, mode=code_path_mode)

    for param_index, param_vals_inst in enumerate(itertools.product(*param_vals)) :


        if only_first_paramtuple and param_index != 0:
            break

        # For each code_path
        for code_path_index, code_path in enumerate(code_paths):

            # Build the base context:
            parameter_values_float = {}
            for (param_index,k) in enumerate(param_syms):
                parameter_values_float[k] = float(param_vals_inst[param_index])

            parameter_values = {}
            for (param_index,k) in enumerate(param_syms):
                parameter_values[k] = float(param_vals_inst[param_index]) * units_dict[k]
            ctx = ActionContext(parameter_values = parameter_values)

            # Execute the code-path:
            code_path.execute(ctx=ctx)

            #Extract the relevant columns and save them to a file:
            column_names = conf['Output Format']['columns']
            basename = conf['Output Format']['base_filename']

            # Get trace data, except time:
            data_col_names = column_names[1:]
            traces = [ctx.res.get_trace(ctx.records[col])for col in data_col_names]
            # Use the time record of the first trace
            time = traces[0].time_pts_ms
            trace_data = [ tr.data_pts/units_dict[col] for col,tr in zip(data_col_names,traces) ]
            trace_data = [tr.rescale(pq.dimensionless).magnitude for tr in trace_data]

            data = [time] + trace_data
            data_matrix = np.vstack(( data) ).T

            opfile = basename.replace('<','{').replace('>','}').format( **parameter_values_float)
            opfile = opfile + 'mfcuke_%d' % code_path_index
            opdir = os.path.join( simtest_utils.Locations.output_root(), conf['scenario_short'])
            opfile = os.path.join(opdir, opfile)

            np.savetxt(opfile, data_matrix)

            if plot_results:
                mf.TagViewer(ctx.res, show=False)
            #break

            #pylab.show()








def main():
    src_files =  glob.glob( simtest_utils.Locations.scenario_descriptions() + "/*.txt")
    skipped =[]
    for fname in src_files:
        run_scenario_filename(fname, only_first_paramtuple=True, plot_results = True)


    print 'Skipped:'
    for fname in skipped:
        print '  ', fname


    pylab.show()


if __name__ == '__main__':
    main()



