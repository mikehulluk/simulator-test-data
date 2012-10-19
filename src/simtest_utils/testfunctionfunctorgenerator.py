
import numpy as np
import scipy.integrate as integrate

import testfunctionparser



class EPSEquality(object):
    def __init__(self, eps_str):
        self.eps_str = eps_str
        self.conds = []

        self.abs_err = None
        self.rel_err = None
        toks = self.eps_str.split('or')
        for t in toks:
            if '%' in t:
                assert not self.rel_err
                self.rel_err = float(t.replace("%",'') ) /100.
            else:
                assert not self.abs_err
                self.abs_err = float(t)
        assert self.abs_err is not None

        

    def compare(self, exp, got):
        """Returns True if they are the same value"""

        # Is the absolute tolerance satisfied:
        if np.fabs( exp-got) < self.abs_err:
            return True

        # Is the relative tolerance satisfied:
        if self.rel_err is not None:
            dx= np.fabs(exp*self.rel_err)
            if ( exp-dx < got < exp+dx ):
                return  True
        return False

    def __str__(self):
        return 'eps: %s %s' % ( self.abs_err, ( '( or %d%%)' % round(self.rel_err*100.) if self.rel_err is not None else '') )


class TableTestFunctor(object):
    def __init__(self, test_expr, expected_value, eps):
        self.test_expr=test_expr
        self.expected_value = expected_value
        self.eps = EPSEquality(eps)

        # Build the data object
        self.condition_info = testfunctionparser.parse_expr(test_expr)

    def to_str(self):
        return '%s == %f (eps:%f)' % (self.test_expr, self.expected_value, self.eps)

    def check_data(self, data_matrix, colnames):
        return self.__call__(data_matrix=data_matrix, colnames=colnames)

    def __call__(self, data_matrix, colnames):
        data_slice = self._get_data_slice(data_matrix=data_matrix, colnames=colnames)
        result = self._apply_operation(data_slice)
        abserr = np.fabs( result - self.expected_value)
        pcerr = ( abserr / self.expected_value ) * 100.
        msg =  '%s ==> %f == %f (%f %f%%) [Res==Expected eps:%s)' % (self.test_expr, result, self.expected_value, abserr,pcerr,  str(self.eps) )
        return self.eps.compare(exp=self.expected_value, got=result), msg, result 
        #return np.fabs(result - self.expected_value) < self.eps, msg, result 


    def _get_data_slice(self, data_matrix, colnames):
        dt = 0.00000001

        var_name = self.condition_info.src_variable
        #print self.condition_info
        assert var_name in colnames, "Can't find variable: %s" % var_name
        col_index = colnames.index(var_name)

        # Get the time and data:
        data_raw = data_matrix[:, (0,col_index)] 
        time_start, time_stop = self.condition_info.slice_

        # Interpolate the data to include start and end points of the slice:
        interp_pts = [ t for t in (time_start,time_stop) if t is not None]
        #Make sure we don't duplicate points if they are already in there:
        interp_pts = [ t for t in interp_pts if np.fabs(data_matrix[:,0]-t).min() > dt ] #
        for p in interp_pts:
            assert data_raw[0,0] <= p <= data_raw[-1,0], 'Slice Indices fall outside of data range: %f [%f %f]' % (p, data_raw[0,0], data_raw[-1,0] )
        if interp_pts:
            # Interpolate the values at the slice points, add them to 
            # the top of the matrix,and sort the matrix by '0' column (time)
            interp_pts = np.array(interp_pts)
            interp_vals = np.interp(interp_pts, data_raw[:,0], data_raw[:,1])
            new_rows = np.vstack( (interp_pts, interp_vals) ).T
            data_raw = np.vstack( (new_rows, data_raw) )
            data_raw = data_raw[data_raw[:,0].argsort()]

        keep_mask = np.empty(data_raw.shape[0], dtype=np.dtype('bool'))
        keep_mask.fill(True)
        if time_start is not None:
            keep_mask = np.logical_and(keep_mask, (data_raw[:,0] >= time_start-dt) )
        if time_stop is not None:
            keep_mask = np.logical_and(keep_mask, (data_raw[:,0] <= time_stop+dt) )


        data_slice = data_raw[keep_mask,:]
        assert data_slice.shape[1] == 2
        assert data_slice.shape[0] >= 2

        return data_slice


    def _apply_operation(self, data_slice):
        function, params = self.condition_info.function

        if function is None:
            raise NotImplementedError('Check over all time (need to change functions slightly!)')

        if function == 'min':
            return float( np.min(data_slice[:,1]) )
        if function == 'max':
            return float( np.max(data_slice[:,1]) )
        if function == 'mean':
            # We have irregularly spaced datam, so calculate the area:
            area = integrate.simps(data_slice[:,1], data_slice[:,0])
            t_range = data_slice[-1,0] - data_slice[0,0]
            res = float( area/t_range )
            return res

        if function == 'at':
            t = params[0]
            res = np.interp([t], data_slice[:,0], data_slice[:,1])
            return float( res[0] )

        assert False, 'Unexpected function found: %s' % function







