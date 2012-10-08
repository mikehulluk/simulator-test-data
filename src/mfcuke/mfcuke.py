

import glob
import configobj
import re


import morphforge.stdimports as mf
import morphforgecontrib.stdimports as mfc
#from morphforge.stdimports import units as U
import pylab
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


class ActionHandle(object):
    def __init__(self, src_regex_str, handlers=None):

        self.src_regex_str = src_regex_str
        self.src_regex = re.compile(src_regex_str)
        self.handlers = handlers or []

    def __str__(self):
        print 'Handler for: %s', self.src_regex_str

    def can_handle(self, expr):
        return self.src_regex.match(expr)

    def __call__(self, ctx, expr ):
        if len(self.handlers) == 1:
            m = self.src_regex.match(expr)
            return self.handlers[0](ctx, expr, groupdict=m.groupdict())
        else:
            print ' ** No handler found! ** '


class ActionHandlerLibrary(object):
    def __init__(self):
        self.handlers =[]

    def register_handler(self, handler):
        self.handlers.append(handler)

    def find_handlers(self, expr):
        return [ h for h in self.handlers if h.can_handle(expr)]
    def get_handler(self, expr):
        handlers = self.find_handlers(expr)
        assert len(handlers) == 1
        return handlers[0]





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
    def new_func(ctx, expr, groupdict, ):
        assert set(groupdict.keys()) == set(['duration'])
        sim_duration = NeuroUnitParser.QuantitySimple(  groupdict['duration']).as_quantities_quantity()
        return func(ctx, sim_duration=sim_duration)
    return new_func

@sim_builder_wrapper
def sim_builder_a(ctx, sim_duration):
    ctx.env = mf.NEURONEnvironment()
    ctx.sim = ctx.env.Simulation(tstop=sim_duration)
# ====================================


# =====================================
def sim_run_wrapper( func ):
    def new_func(ctx, expr, groupdict):
        assert not groupdict
        return func(ctx,)
    return new_func

@sim_run_wrapper
def sim_run(ctx):
    ctx.res = ctx.sim.run()
    print ctx.res.get_traces()
    for recname, recobj in ctx.records.items():
        print recname, ctx.res.get_trace(recobj)
# =====================================



# =====================================
def sim_build_single_compartment_wrapper( func ):
    def new_func(ctx, expr, groupdict, ):
        assert set(groupdict.keys()) == set(['name','params'])
        name = groupdict['name']

        # Break the parameter-string into component parts:
        params = parse_param_str(groupdict['params'])
        # Resolve possible variables:
        params = ctx.resolve_context_parameter_values(params)

        return func(ctx,
                name=name,
                initialvoltage= params.get('initialvoltage',None),
                capacitance = params.get('capacitance',None),
                area=params.get('area',None))

    return new_func

@sim_build_single_compartment_wrapper
def sim_build_single_compartment(ctx, name, initialvoltage, capacitance, area ):
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
    def new_func(ctx, expr, groupdict):
        assert set(groupdict.keys()) == set(['where','what','as'])
        where = groupdict['where']
        what = groupdict['what']
        as_ = groupdict['as']
        return func(ctx,where=where, what=what,as_=as_)
    return new_func

@sim_record_wrapper
def sim_record(ctx, where, what, as_):
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
    def new_func(ctx, expr, groupdict):
        assert set(groupdict.keys()) == set(['amplitude','cell','location', 'from','until'])

        # Resolve the groupdict into units for 'amp','for' and 'until'
        groupdict = ctx.resolve_context_parameter_values(groupdict)
        for k,v in groupdict.items():
            if not k in ['amplitude','from','until']:
                continue
            if isinstance(v, basestring):
                groupdict[k] = NeuroUnitParser.QuantitySimple(v).as_quantities_quantity()

        return func(ctx,
            amplitude = groupdict['amplitude'],
            from_ = groupdict['from'],
            until = groupdict['until'],
            cell = groupdict['cell'],
            location = groupdict['location'] or 'soma'
            )
    return new_func

@sim_step_current_injection_wrapper
def sim_step_current_injection(ctx, amplitude, from_, until, cell, location):
    cell = ctx.obj_refs[cell]
    ctx.sim.create_currentclamp(cell_location=cell.get_location(location),
                                amp=amplitude,
                                delay=from_,
                                dur=until-from_)
# =====================================


# =====================================
def sim_add_channel_wrapper( func ):
    def new_func(ctx, expr, groupdict):
        assert set(groupdict.keys()) == set(['channelname','cell','params'])
        # Break the parameter-string into component parts:
        params = parse_param_str(groupdict['params'])
        # Resolve possible variables:
        params = ctx.resolve_context_parameter_values(params)
        return func(ctx, 
                channelname=groupdict['channelname'],
                cell_name=groupdict['cell'],
                **params)
    return new_func

@sim_add_channel_wrapper
def sim_add_channel(ctx, channelname, cell_name, **kwargs):

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

handler_lib.register_handler( ActionHandle( src_regex_str = r"""In a simulation lasting (?P<duration>[0-9.]*ms)""", handlers=[sim_builder_a]) )
handler_lib.register_handler( ActionHandle( src_regex_str = r"""Run the simulation""", handlers=[sim_run]))
handler_lib.register_handler( ActionHandle( src_regex_str = r"""Create a single compartment neuron '(?P<name>[a-zA-Z0-9_]+)' with (?P<params>.*)""", handlers=[sim_build_single_compartment] ) )
handler_lib.register_handler( ActionHandle( src_regex_str = r"""Record (?P<where>[a-zA-Z0-9_]*)\.(?P<what>[a-zA-Z0-9_]*) as (?P<as>[a-zA-Z0-9_$]*)""", handlers=[sim_record]))
handler_lib.register_handler( ActionHandle( src_regex_str = r"""Inject step-current of (?P<amplitude>.*) into (?P<cell>[a-zA-Z0-9]*)(\.(?P<location>[a-zA-Z0-9]*))? from t=(?P<from>.*) until t=(?P<until>.*)""", handlers=[sim_step_current_injection]) )
handler_lib.register_handler( ActionHandle( src_regex_str = r"""Add (?P<channelname>[a-zA-Z0-9_]+)* channels to (?P<cell>[a-zA-Z0-9_]+)* with (?P<params>.*)""", handlers=[sim_add_channel]) )


handler_lib.register_handler( ActionHandle( src_regex_str = r"""Create an gap junction '[a-zA-Z0-9_]*'  with resistance <RGJ1> between ([a-zA-Z0-9_]*) and ([a-zA-Z0-9_]*)""") )
handler_lib.register_handler( ActionHandle( src_regex_str = r"""Create a .* synapse onto cell1 with (.*) driven with spike-times at \[(.*)\]""") )








def run_scenario_filename(fname):
    conf = configobj.ConfigObj(fname)

    parameter_values = {'A': mf.unit('10000:um2'),
                        'C': mf.unit('1.0:uF/cm2'),
                        'I': mf.unit('120:pA'),
                        'GLK': mf.unit('0.3:mS/cm2'),
                        'VS': mf.unit('-31:mV'),
                        'EREV': mf.unit('-31:mV') }

    ctx = ActionContext(parameter_values = parameter_values)

    for exec_cmd in conf['description'].split("\n"):
        exec_cmd = exec_cmd.strip()

        if not exec_cmd:
            continue

        print exec_cmd
        handler = handler_lib.get_handler(exec_cmd)
        handler.__call__(ctx, exec_cmd)
        print

    
    mf.TagViewer(ctx.res)
    pylab.show()







src_dir = '../../scenario_descriptions/'
src_files = glob.glob(src_dir + '*.txt')

skipped =[]
for fname in src_files:
    if not 'scenario001' in fname:
        continue
    run_scenario_filename(fname)


print 'Skipped:'
for fname in skipped:
    print '  ', fname






