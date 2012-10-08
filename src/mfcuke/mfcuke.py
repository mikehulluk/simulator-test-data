

import glob
import configobj
import re
import itertools
import pylab
import os
import numpy as np

import quantities as pq
import morphforge.stdimports as mf
import morphforgecontrib.stdimports as mfc
from neurounits import NeuroUnitParser


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
        assert len(handlers) == 1
        return handlers[0]

    def build_code_paths(self, description_lines):

        # Build a list of all possible functors for each line:
        all_child_handlers = []
        for i, line in enumerate(description_lines):
            print line
            h = self.get_parent_handler(line)
            all_child_handlers.append( h.child_handlers )

        # Take the outer-product to get all the code-paths:
        code_paths = []
        for hs in itertools.product(*all_child_handlers):
            code_paths.append(ActionCodePath(description_lines, hs) )
        return code_paths








def parse_param_str(expr):
    res = {}
    for param_tok in re.split(r"""(?:and)|(?:,)""", expr):
        p0,p1 = param_tok.strip().split()
        assert not p0 in res
        res[p0] = p1
    return res

def is_context_parameter(expr):
    if not expr:
        return
    m = re.match(r"""\<(?P<name>.*)\>""", expr)
    if not m:
        return None
    return m.groupdict()['name']







# =====================================
def sim_builder_wrapper( func ):
    def new_func(self, ctx, expr, groupdict, ):
        assert set(groupdict.keys()) == set(['duration'])
        sim_duration = NeuroUnitParser.QuantitySimple(  groupdict['duration']).as_quantities_quantity()
        return func(self, ctx, sim_duration=sim_duration)
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

        # Break the parameter-string into component parts:
        params = parse_param_str(groupdict['params'])
        # Resolve possible variables:
        params = ctx.resolve_context_parameter_values(params)

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
        for k,v in groupdict.items():
            if not k in ['amplitude','from','until']:
                continue
            if isinstance(v, basestring):
                groupdict[k] = NeuroUnitParser.QuantitySimple(v).as_quantities_quantity()

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
            chl = ctx.env.Channel(mfc.StdChlLeak, conductance=kwargs['conductance'], reversalpotential=kwargs['reversalpotential'] )
        else:
            assert False

        # Apply the channel:
        cell = ctx.obj_refs[cell_name]
        cell.apply_channel(chl)
# =====================================










handler_lib = ActionHandlerLibrary()

handler_lib.register_handler( ActionHandleParent( src_regex_str = r"""In a simulation lasting (?P<duration>[0-9.]*ms)""", child_handlers=[sim_builder_a(),sim_builder_b()]) )
handler_lib.register_handler( ActionHandleParent( src_regex_str = r"""Run the simulation""", child_handlers=[sim_run()]))
handler_lib.register_handler( ActionHandleParent( src_regex_str = r"""Create a single compartment neuron '(?P<name>[a-zA-Z0-9_]+)' with (?P<params>.*)""", child_handlers=[sim_build_single_compartment()] ) )
handler_lib.register_handler( ActionHandleParent( src_regex_str = r"""Record (?P<where>[a-zA-Z0-9_]*)\.(?P<what>[a-zA-Z0-9_]*) as [$](?P<as>[a-zA-Z0-9_$]*)""", child_handlers=[sim_record()]))
handler_lib.register_handler( ActionHandleParent( src_regex_str = r"""Inject step-current of (?P<amplitude>.*) into (?P<cell>[a-zA-Z0-9]*)(\.(?P<location>[a-zA-Z0-9]*))? from t=(?P<from>.*) until t=(?P<until>.*)""", child_handlers=[sim_step_current_injection()]) )
handler_lib.register_handler( ActionHandleParent( src_regex_str = r"""Add (?P<channelname>[a-zA-Z0-9_]+)* channels to (?P<cell>[a-zA-Z0-9_]+)* with (?P<params>.*)""", child_handlers=[sim_add_channel()]) )


handler_lib.register_handler( ActionHandleParent( src_regex_str = r"""Create a gap junction '[a-zA-Z0-9_]*'  with resistance <RGJ1> between ([a-zA-Z0-9_]*) and ([a-zA-Z0-9_]*)""") )
handler_lib.register_handler( ActionHandleParent( src_regex_str = r"""Create a .* synapse onto cell1 with (.*) driven with spike-times at \[(.*)\]""") )








def run_scenario_filename(fname):
    conf = configobj.ConfigObj(fname)
    


    units_dict = dict( [(k,NeuroUnitParser.Unit(v).as_quantities_unit() ) for (k,v) in conf['Units'].iteritems() ] )
    param_syms = conf['Parameter Values'].keys()
    param_vals = [ conf['Parameter Values'][sym] for sym in param_syms]



    description_lines = [ l.strip() for l in conf['description'].split('\n')]
    description_lines = [ l for l in description_lines if l]

    code_paths = handler_lib.build_code_paths(description_lines)


    for param_index, param_vals_inst in enumerate(itertools.product(*param_vals)) :
        #if param_index > 1:
        #    continue

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
            opdir = os.path.join(output_dir, conf['scenario_short'])
            opfile = os.path.join(opdir, opfile)

            np.savetxt(opfile, data_matrix)





            #mf.TagViewer(ctx.res, show=False)




    pylab.show()







src_dir = '../../scenario_descriptions/'
output_dir = '../../output/'
src_files = glob.glob(src_dir + '*.txt')

skipped =[]
for fname in src_files:
    if not 'scenario001' in fname:
        continue
    run_scenario_filename(fname)


print 'Skipped:'
for fname in skipped:
    print '  ', fname






