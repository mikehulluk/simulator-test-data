

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




def configure(ctx):
    pass

def ensure_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



def ensure_directory_structure_setup(ctx):
    # Make sure the output folders are setup:
    ensure_exists('output')
    ensure_exists('output/scenario001')
    ensure_exists('output/scenario020')
    ensure_exists('output/scenario021')
    ensure_exists('output/scenario075')


    # Look in all the folders in 'simulators' and make sure that thier 'output'
    # directories point to the right place:
    re_dirname = re.compile(r"""^simulators/(?P<sim>[a-zA-Z0-9_]*)/(?P<scen>[a-zA-Z0-9_]*)$""")
    for name in glob2.glob('simulators/**/*'):
        if not os.path.isdir(name):
            continue
        m = re_dirname.match(name)
        if not m:
            continue

        print name
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





def generate(ctx):
    
    ensure_directory_structure_setup(ctx)
    
    # Call the simulators:
    ctx.recurse('simulators')

def cleanup(ctx):
    shutil.rmtree('output')
    

def compare(ctx):
    import simtest_utils
    simtest_utils.check_all_scenarios()
