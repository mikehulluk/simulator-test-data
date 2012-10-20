

# Hook in the local 'src' dir onto our pythonm path
import sys
import os
local_dir = os.getcwd()
local_src = os.path.join(local_dir, 'src')
sys.path.append(local_src)

glob2_src = os.path.join(local_dir, 'src/glob2/src')
sys.path.append(glob2_src)
import glob2

import re
import shutil


import mredoc as mrd


def configure(ctx):
    pass

def ensure_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



def ensure_directory_structure_setup(ctx):

    import waf_util

     # Make sure the output folders are setup:
    ensure_exists('output')
    for scen in waf_util.get_all_scenarios():
        ensure_exists('output/%s'%scen)

    # Look in all the folders in 'simulators' and make sure that thier 'output'
    # directories point to the right place:
    re_dirname = re.compile(r"""^simulators/(?P<sim>[a-zA-Z0-9_]*)/(?P<scen>[a-zA-Z0-9_]*)$""")
    for name in glob2.glob('simulators/**/*'):
        if not os.path.isdir(name):
            continue
        m = re_dirname.match(name)
        if not m:
            continue

        sim, scen = m.groupdict()['sim'],  m.groupdict()['scen']
        oplink = os.path.join( name, 'output')
        opdir = os.path.abspath( os.path.join('output', scen) )
        assert os.path.exists(opdir)

        if os.path.lexists(oplink):
            print 'Already Linked'
            # The link exists, but is it broken?
            assert os.path.exists(os.readlink(oplink)),'Output Directory not found, although link exists: %s to %s'%(oplink, os.readlink(oplink))

            # Does it point to the right place?
            assert os.path.samefile(os.readlink(oplink), opdir),''
        else:
            print "Linking: '%s' -> '%s' " % (oplink, opdir)
            assert not os.path.exists(oplink)
            assert not os.path.lexists(oplink)
            os.symlink(opdir, oplink)







def produce_mredoc_output(sim_details,sim_overview,mrd_gen=None):
    # Generate the output:
    sect = mrd.Section('Simulator Comparison',
                        mrd_gen,
                        sim_overview,
                        sim_details
                        )

    sect.to_html('~/test_results/')
    #sect.to_pdf('~/test_results/output.pdf')



def setupdirs(ctx):
    ensure_directory_structure_setup(ctx)

def generate(ctx):

    ensure_directory_structure_setup(ctx)

    return ctx.recurse('simulators', name='generate')




def cleanup(ctx):
    if os.path.exists('output'):
        shutil.rmtree('output')
    ctx.recurse('simulators/10_neuron/scenario075', name='cleanup')


def compare(ctx):


    (sim_details,sim_overview) = _compare(ctx=ctx)

    # Generate output:
    produce_mredoc_output(
        sim_details=sim_details,
        sim_overview=sim_overview)



def _compare(ctx):
    import simtest_utils
    return simtest_utils.check_scenarios(create_mredoc=True)


def all(ctx):
    cleanup(ctx)

    # Generate the results files:
    generate(ctx)
    mrd_gen = ctx.sim_generate_redoc

    # Compare the outputs:
    (sim_details,sim_overview) = _compare(ctx)

    # Generate the output:
    produce_mredoc_output(
        sim_details=sim_details,
        sim_overview=sim_overview,
        mrd_gen=mrd_gen,
        )



