import os

def chdirdecorator(func):
    def new_func(ctx, *args, **kwargs):
        print 'Changing dir'
        old_loc = os.getcwd()
        os.chdir( ctx.path.abspath() ) 
        res = func(ctx, *args, **kwargs)
        os.chdir(old_loc)
        return res
    return new_func
