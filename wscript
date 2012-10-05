

# Hook in the local 'src' dir onto our pythonm path
import sys
import os
local_dir = os.getcwd()
local_src = os.path.join(local_dir, 'src')
sys.path.append(local_src)



def configure(ctx):
    pass

def ensure_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def generate(ctx):

    # Make sure the output folders are setup:
    ensure_exists('output')
    ensure_exists('output/scenario001')
    ensure_exists('output/scenario020')

    # Call the simulators:
    ctx.recurse('simulators')

def cleanup(ctx):
    ctx.recurse('simulators')



def compare(ctx):
    import simtest_utils
    simtest_utils.check_all_scenarios()
