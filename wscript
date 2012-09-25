

# Hook in the local 'src' dir onto our pythonm path
import sys
import os
local_dir = os.getcwd()
local_src = os.path.join(local_dir, 'src')
sys.path.append(local_src)



def configure(ctx):
    pass


def generate(ctx):
    ctx.recurse('simulators')

def cleanup(ctx):
    ctx.recurse('simulators')



def compare(ctx):
    import simtest_utils
    simtest_utils.check_all_scenarios()
