

import ply.lex
import ply.yacc



test_data = [
    'V',
    'V[3:50]',
    'V[5:10].min',
]

class TableOutputExpr(object):
    def __init__(self, src_variable, slice_, function):
        self.src_variable = src_variable
        self.slice_ = slice_
        self.function = function

    def __str__(self):
        V = self.src_variable
        S = '[%s:%s]'% ( (self.slice_[0] if self.slice_[0] is not None else ''), (self.slice_[1] if self.slice_[1] is not None else '') )
        F = '.%s' % self.function if self.function is not None else ''
        return '%s%s%s' % (V, S, F)

    def to_str(self):
        return str(self)





# Define the LEX 
# --------------------
tokens = (
   'ID',
   'NUMBER',
   'LSQBRACKET',
   'RSQBRACKET',
   'COLON',
   'DOT',
)

# Regular expression rules for simple tokens
t_LSQBRACKET  = r'\['
t_RSQBRACKET  = r'\]'
t_COLON  = r':'
t_DOT  = r'\.'
t_ID = """[A-Za-z0-9_]+"""
t_ignore  = ' \t'


def t_NUMBER(t):
    """(\+|-)?([0-9]+\.?[0-9]*|\.[0-9]+)([eE](\+|-)?[0-9]+)?"""
    t.value = float(t.value)
    return t

def t_error(t):
        print "Illegal character '%s'" % t.value[0]
        assert False
        t.lexer.skip(1)




# Define the YACC
# ---------------

def p_expression_output_expr1(p):
    'output_expr : array_section '
    p[0] = TableOutputExpr(src_variable=p[1][0],
                           slice_ = p[1][1],
                           function = [None,None])

def p_expression_output_expr2(p):
    'output_expr : array_section DOT function_name'
    p[0] = TableOutputExpr(src_variable=p[1][0],
                           slice_ = p[1][1],
                           function = [p[3], None])

def p_expression_output_expr3(p):
    'output_expr : array_section LSQBRACKET NUMBER RSQBRACKET'
    p[0] = TableOutputExpr(src_variable=p[1][0],
                           slice_ = p[1][1],
                           function = ['at', p[3] ])

def p_expression_array_name(p):
    'array_name : ID'
    p[0] = p[1]

def p_expression_func_name(p):
    'function_name : ID'
    p[0] = p[1]

def p_expression_arraysection1(p):
    'array_section : array_name'
    p[0] = ( p[1], [None,None] )


def p_empty(p):
    'empty : '
    p[0] = None

def p_slice_number(p):
    """slice_number : NUMBER
                    | empty"""
    p[0] = p[1]

def p_expression_arraysection2(p):
    'array_section : array_name LSQBRACKET slice_number COLON slice_number RSQBRACKET'
    p[0] = ( p[1], (p[3],p[5]) )




lexer = ply.lex.lex()
parser = ply.yacc.yacc(outputdir="somedirectory", write_tables=0)


def parse_expr(s):
    return parser.parse(s, lexer=lexer)


if __name__ == '__main__':
    for t in test_data:
        res = parser.parse(t, lexer=lexer)
        print res

