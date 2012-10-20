import os
import glob
import fnmatch
import configobj

from simtest_utils import Locations


def chdirdecorator(func):
    def new_func(ctx, *args, **kwargs):
        old_loc = os.getcwd()
        print 'Changing dir to:', ctx.path.abspath()
        os.chdir( ctx.path.abspath() )
        res = func(ctx, *args, **kwargs)
        os.chdir(old_loc)
        return res
    return new_func

def rmtree(location):
    for root, dirs, files in os.walk(location, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def ensure_output_links_setup():
    """ Assumes that we are in the directory already """
    assert False, 'Deprecated'

    return
    scenario_name = os.path.split( os.getcwd() )[-1]
    if not os.path.exists('output'): #and not os.path.lexists('output/'):
        os.symlink('../../../output/%s/' %scenario_name, 'output' )
    if not os.path.exists('output/'):
        raise IOError("Can't find file: %s" %'')








def get_all_simulators():
    simulators = {
        'NEURON':'10_neuron',
        'morphforge':'20_morphforge',
        'mfcuke':'30_mfcuke',
    }
    return simulators

def get_all_scenarios():
    """returns a dict 'name'->'filename'"""
    dct = {}
    for scenario_file in glob.glob(Locations.scenario_descriptions() + '/*.txt'):
        print scenario_file
        name = configobj.ConfigObj(scenario_file)['scenario_short']

        # Sanity checking the filename:
        fname_short = os.path.split(scenario_file)[-1]
        assert fname_short.startswith(name), 'Inconstistent naming: %s %s' %( fname_short, name)

        assert not name in dct
        dct[name] = scenario_file

    return dct


def get_target_simulator_dirs():

    simulator_lut = get_all_simulators()

    simulator_str = os.environ.get('STD_SIMS','*')
    simulator_str = simulator_str.strip()

    dirs_to_recurse = []
    for tok in simulator_str.split(';'):
        if tok == '*':
            dirs_to_recurse.extend( simulator_lut.values() )
        else:
            dirs_to_recurse.append( simulator_lut[tok] )
    return sorted(set(dirs_to_recurse))


def get_target_scenarios():
    scen_str = os.environ.get('STD_SCENS','*')
    scen_str = scen_str.strip()

    all_scenarios = get_all_scenarios()

    scens = set()
    for tok in scen_str.split(';'):

        scens.update(fnmatch.filter(all_scenarios, tok))
        scens.update(fnmatch.filter(all_scenarios, 'scenario' + tok))

    return sorted(scens)


def is_short_run():
    if 'STD_SHORT' in os.environ:
        return True
    return False

    return True

