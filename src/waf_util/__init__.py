import os

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
        #for name in dirs:
        #    os.rmdir(os.path.join(root, name))
