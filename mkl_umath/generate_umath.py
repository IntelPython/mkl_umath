import os
import re
import struct
import sys
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
import ufunc_docstrings as docstrings
sys.path.pop(0)

Zero = "PyLong_FromLong(0)"
One = "PyLong_FromLong(1)"
True_ = "(Py_INCREF(Py_True), Py_True)"
False_ = "(Py_INCREF(Py_False), Py_False)"
None_ = object()
AllOnes = "PyInt_FromLong(-1)"
MinusInfinity = 'PyFloat_FromDouble(-NPY_INFINITY)'
ReorderableNone = "(Py_INCREF(Py_None), Py_None)"

# Sentinel value to specify using the full type description in the
# function name
class FullTypeDescr:
    pass

class FuncNameSuffix:
    """Stores the suffix to append when generating functions names.
    """
    def __init__(self, suffix):
        self.suffix = suffix

class TypeDescription:
    """Type signature for a ufunc.

    Attributes
    ----------
    type : str
        Character representing the nominal type.
    func_data : str or None or FullTypeDescr or FuncNameSuffix, optional
        The string representing the expression to insert into the data
        array, if any.
    in_ : str or None, optional
        The typecode(s) of the inputs.
    out : str or None, optional
        The typecode(s) of the outputs.
    astype : dict or None, optional
        If astype['x'] is 'y', uses PyUFunc_x_x_As_y_y/PyUFunc_xx_x_As_yy_y
        instead of PyUFunc_x_x/PyUFunc_xx_x.
    simd: list
        Available SIMD ufunc loops, dispatched at runtime in specified order
        Currently only supported for simples types (see make_arrays)
    """
    def __init__(self, type, f=None, in_=None, out=None, astype=None, simd=None):
        self.type = type
        self.func_data = f
        if astype is None:
            astype = {}
        self.astype_dict = astype
        if in_ is not None:
            in_ = in_.replace('P', type)
        self.in_ = in_
        if out is not None:
            out = out.replace('P', type)
        self.out = out
        self.simd = simd

    def finish_signature(self, nin, nout):
        if self.in_ is None:
            self.in_ = self.type * nin
        assert len(self.in_) == nin
        if self.out is None:
            self.out = self.type * nout
        assert len(self.out) == nout
        self.astype = self.astype_dict.get(self.type, None)

_fdata_map = dict(
    e='npy_%sf',
    f='npy_%sf',
    d='npy_%s',
    g='npy_%sl',
    F='nc_%sf',
    D='nc_%s',
    G='nc_%sl'
)

def build_func_data(types, f):
    func_data = [_fdata_map.get(t, '%s') % (f,) for t in types]
    return func_data

def TD(types, f=None, astype=None, in_=None, out=None, simd=None):
    if f is not None:
        if isinstance(f, str):
            func_data = build_func_data(types, f)
        elif len(f) != len(types):
            raise ValueError("Number of types and f do not match")
        else:
            func_data = f
    else:
        func_data = (None,) * len(types)
    if isinstance(in_, str):
        in_ = (in_,) * len(types)
    elif in_ is None:
        in_ = (None,) * len(types)
    elif len(in_) != len(types):
        raise ValueError("Number of types and inputs do not match")
    if isinstance(out, str):
        out = (out,) * len(types)
    elif out is None:
        out = (None,) * len(types)
    elif len(out) != len(types):
        raise ValueError("Number of types and outputs do not match")
    tds = []
    for t, fd, i, o in zip(types, func_data, in_, out):
        # [(simd-name, list of types)]
        if simd is not None:
            simdt = [k for k, v in simd if t in v]
        else:
            simdt = []
        tds.append(TypeDescription(t, f=fd, in_=i, out=o, astype=astype, simd=simdt))
    return tds

class Ufunc:
    """Description of a ufunc.

    Attributes
    ----------
    nin : number of input arguments
    nout : number of output arguments
    identity : identity element for a two-argument function
    docstring : docstring for the ufunc
    type_descriptions : list of TypeDescription objects
    """
    def __init__(self, nin, nout, identity, docstring, typereso,
                 *type_descriptions, signature=None):
        self.nin = nin
        self.nout = nout
        if identity is None:
            identity = None_
        self.identity = identity
        self.docstring = docstring
        self.typereso = typereso
        self.type_descriptions = []
        self.signature = signature
        for td in type_descriptions:
            self.type_descriptions.extend(td)
        for td in self.type_descriptions:
            td.finish_signature(self.nin, self.nout)

# String-handling utilities to avoid locale-dependence.

import string
UPPER_TABLE = bytes.maketrans(bytes(string.ascii_lowercase, "ascii"),
                              bytes(string.ascii_uppercase, "ascii"))

def english_upper(s):
    """ Apply English case rules to convert ASCII strings to all upper case.

    This is an internal utility function to replace calls to str.upper() such
    that we can avoid changing behavior with changing locales. In particular,
    Turkish has distinct dotted and dotless variants of the Latin letter "I" in
    both lowercase and uppercase. Thus, "i".upper() != "I" in a "tr" locale.

    Parameters
    ----------
    s : str

    Returns
    -------
    uppered : str

    Examples
    --------
    >>> from numpy.lib.utils import english_upper
    >>> s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_'
    >>> english_upper(s)
    'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    >>> english_upper('')
    ''
    """
    uppered = s.translate(UPPER_TABLE)
    return uppered


#each entry in defdict is a Ufunc object.

#name: [string of chars for which it is defined,
#       string of characters using func interface,
#       tuple of strings giving funcs for data,
#       (in, out), or (instr, outstr) giving the signature as character codes,
#       identity,
#       docstring,
#       output specification (optional)
#       ]

chartoname = {
    '?': 'bool',
    'b': 'byte',
    'B': 'ubyte',
    'h': 'short',
    'H': 'ushort',
    'i': 'int',
    'I': 'uint',
    'l': 'long',
    'L': 'ulong',
    'q': 'longlong',
    'Q': 'ulonglong',
    'e': 'half',
    'f': 'float',
    'd': 'double',
    'g': 'longdouble',
    'F': 'cfloat',
    'D': 'cdouble',
    'G': 'clongdouble',
    'M': 'datetime',
    'm': 'timedelta',
    'O': 'OBJECT',
    # '.' is like 'O', but calls a method of the object instead
    # of a function
    'P': 'OBJECT',
}

noobj = '?bBhHiIlLqQefdgFDGmM'
all = '?bBhHiIlLqQefdgFDGOmM'

O = 'O'
P = 'P'
ints = 'bBhHiIlLqQ'
times = 'Mm'
timedeltaonly = 'm'
intsO = ints + O
bints = '?' + ints
bintsO = bints + O
flts = 'efdg'
fltsO = flts + O
fltsP = flts + P
cmplx = 'FDG'
cmplxvec = 'FD'
cmplxO = cmplx + O
cmplxP = cmplx + P
inexact = flts + cmplx
inexactvec = 'fd'
noint = inexact+O
nointP = inexact+P
allP = bints+times+flts+cmplxP
nobool_or_obj = noobj[1:]
nobool_or_datetime = noobj[1:-1] + O # includes m - timedelta64
intflt = ints+flts
intfltcmplx = ints+flts+cmplx
nocmplx = bints+times+flts
nocmplxO = nocmplx+O
nocmplxP = nocmplx+P
notimes_or_obj = bints + inexact
nodatetime_or_obj = bints + inexact

# Find which code corresponds to int64.
int64 = ''
uint64 = ''
for code in 'bhilq':
    if struct.calcsize(code) == 8:
        int64 = code
        uint64 = english_upper(code)
        break

# This dictionary describes all the ufunc implementations, generating
# all the function names and their corresponding ufunc signatures.  TD is
# an object which expands a list of character codes into an array of
# TypeDescriptions.
defdict = {
'add':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.add'),
          None,
          TD(inexactvec + cmplxvec),
          ),
'subtract':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.subtract'),
          None,
          TD(inexactvec + cmplxvec),
          ),
'multiply':
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.multiply'),
          None, 
          TD(inexactvec + cmplxvec),
          ),
'floor_divide':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.floor_divide'),
          None,
          TD(inexactvec + cmplxvec),
          ),
'true_divide':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.true_divide'),
          None,
          TD(inexactvec + cmplxvec),
          ),
'conjugate':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.conjugate'),
          None,
          TD(inexactvec + cmplxvec),
          ),
# 'fmod':
'square':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.square'),
          None,
          TD(inexactvec + cmplxvec),
          ),
'reciprocal':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.reciprocal'),
          None,
          TD(inexactvec + cmplxvec),
          ),
# '_ones_like':
# 'power':
# 'float_power':
'absolute':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.absolute'),
          None,
          TD(inexactvec),
          TD('F', out='f'),
          TD('D', out='d'),
          ),
# '_arg':
'negative':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.negative'),
          None,
          TD(inexactvec),
          ),
'positive':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.positive'),
          None,
          TD(inexactvec),
          ),
'sign':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sign'),
          None,
          TD(inexactvec + cmplxvec),
          ),
'greater':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.greater'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'greater_equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.greater_equal'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'less':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.less'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'less_equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.less_equal'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.equal'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'not_equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.not_equal'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'logical_and':
    Ufunc(2, 1, True_,
          docstrings.get('numpy.core.umath.logical_and'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'logical_not':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.logical_not'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'logical_or':
    Ufunc(2, 1, False_,
          docstrings.get('numpy.core.umath.logical_or'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'logical_xor':
    Ufunc(2, 1, False_,
          docstrings.get('numpy.core.umath.logical_xor'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'maximum':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.maximum'),
          None,
          TD(inexactvec + cmplxvec),
          ),
'minimum':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.minimum'),
          None,
          TD(inexactvec + cmplxvec),
          ),
# 'clip':
'fmax':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.fmax'),
          None,
          TD(inexactvec + cmplxvec),
          ),
'fmin':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.fmin'),
          None,
          TD(inexactvec + cmplxvec),
          ),
# 'logaddexp':
# 'logaddexp2':
# 'bitwise_and':
# 'bitwise_or':
# 'bitwise_xor':
# 'invert':
# 'left_shift':
# 'right_shift':
# 'heaviside':
# 'degrees':
# 'rad2deg':
# 'radians':
# 'deg2rad':
'arccos':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arccos'),
          None,
          TD(inexactvec),
          ),
'arccosh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arccosh'),
          None,
          TD(inexactvec),
          ),
'arcsin':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arcsin'),
          None,
          TD(inexactvec),
          ),
'arcsinh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arcsinh'),
          None,
          TD(inexactvec),
          ),
'arctan':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arctan'),
          None,
          TD(inexactvec),
          ),
'arctanh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arctanh'),
          None,
          TD(inexactvec),
          ),
'cos':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cos'),
          None,
          TD(inexactvec),
          ),
'sin':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sin'),
          None,
          TD(inexactvec),
          ),
'tan':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.tan'),
          None,
          TD(inexactvec),
          ),
'cosh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cosh'),
          None,
          TD(inexactvec),
          ),
'sinh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sinh'),
          None,
          TD(inexactvec),
          ),
'tanh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.tanh'),
          None,
          TD(inexactvec),
          ),
'exp':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.exp'),
          None,
          TD(inexactvec),
          ),
'exp2':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.exp2'),
          None,
          TD(inexactvec),
          ),
'expm1':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.expm1'),
          None,
          TD(inexactvec),
          ),
'log':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log'),
          None,
          TD(inexactvec),
          ),
'log2':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log2'),
          None,
          TD(inexactvec),
          ),
'log10':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log10'),
          None,
          TD(inexactvec),
          ),
'log1p':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log1p'),
          None,
          TD(inexactvec),
          ),
'sqrt':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sqrt'),
          None,
          TD(inexactvec),
          ),
'cbrt':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cbrt'),
          None,
          TD(inexactvec),
          ),
'ceil':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.ceil'),
          None,
          TD(inexactvec),
          ),
'trunc':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.trunc'),
          None,
          TD(inexactvec),
          ),
'fabs':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.fabs'),
          None,
          TD(inexactvec),
       ),
'floor':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.floor'),
          None,
          TD(inexactvec),
          ),
'rint':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.rint'),
          None,
          TD(inexactvec),
          ),
# 'arctan2':
'remainder':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.remainder'),
          None,
          TD(inexactvec),
          ),
# 'divmod':
# 'hypot':
'isnan':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isnan'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
# 'isnat':
'isinf':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isinf'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
'isfinite':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isfinite'),
          None,
          TD(inexactvec + cmplxvec, out='?'),
          ),
# 'signbit':
'copysign':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.copysign'),
          None,
          TD(inexactvec),
          ),
'nextafter':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.nextafter'),
          None,
          TD(inexactvec),
          ),
'spacing':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.spacing'),
          None,
          TD(inexactvec),
          ),
'modf':
    Ufunc(1, 2, None,
          docstrings.get('numpy.core.umath.modf'),
          None,
          TD(inexactvec),
          ),
'ldexp' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.ldexp'),
          None,
          [
              TypeDescription('f', None, 'fi', 'f'),
              TypeDescription('f', FuncNameSuffix('long'), 'fl', 'f'),
              TypeDescription('d', None, 'di', 'd'),
              TypeDescription('d', FuncNameSuffix('long'), 'dl', 'd'),
          ],
          ),
'frexp' :
    Ufunc(1, 2, None,
          docstrings.get('numpy.core.umath.frexp'),
          None,
          [
              TypeDescription('f', None, 'f', 'fi'),
              TypeDescription('d', None, 'd', 'di'),
          ],
          ),
# 'gcd' :
# 'lcm' :
# 'matmul' :
}

def indent(st, spaces):
    indentation = ' '*spaces
    indented = indentation + st.replace('\n', '\n'+indentation)
    # trim off any trailing spaces
    indented = re.sub(r' +$', r'', indented)
    return indented

# maps [nin, nout][type] to a suffix
arity_lookup = {
    (1, 1): {
        'e': 'e_e',
        'f': 'f_f',
        'd': 'd_d',
        'g': 'g_g',
        'F': 'F_F',
        'D': 'D_D',
        'G': 'G_G',
        'O': 'O_O',
        'P': 'O_O_method',
    },
    (2, 1): {
        'e': 'ee_e',
        'f': 'ff_f',
        'd': 'dd_d',
        'g': 'gg_g',
        'F': 'FF_F',
        'D': 'DD_D',
        'G': 'GG_G',
        'O': 'OO_O',
        'P': 'OO_O_method',
    },
    (3, 1): {
        'O': 'OOO_O',
    }
}

#for each name
# 1) create functions, data, and signature
# 2) fill in functions and data in InitOperators
# 3) add function.

def make_arrays(funcdict):
    # functions array contains an entry for every type implemented NULL
    # should be placed where PyUfunc_ style function will be filled in
    # later
    code1list = []
    code2list = []
    names = sorted(funcdict.keys())
    for name in names:
        uf = funcdict[name]
        funclist = []
        datalist = []
        siglist = []
        k = 0
        sub = 0

        for t in uf.type_descriptions:
            if t.func_data is FullTypeDescr:
                tname = english_upper(chartoname[t.type])
                datalist.append('(void *)NULL')
                funclist.append(
                        '%s_%s_%s_%s' % (tname, t.in_, t.out, name))
            elif isinstance(t.func_data, FuncNameSuffix):
                datalist.append('(void *)NULL')
                tname = english_upper(chartoname[t.type])
                funclist.append(
                        '%s_%s_%s' % (tname, name, t.func_data.suffix))
            elif t.func_data is None:
                datalist.append('(void *)NULL')
                tname = english_upper(chartoname[t.type])
                funclist.append('%s_%s' % (tname, name))
                if t.simd is not None:
                    for vt in t.simd:
                        code2list.append(textwrap.dedent("""\
                        #ifdef HAVE_ATTRIBUTE_TARGET_{ISA}
                        if (NPY_CPU_HAVE({ISA})) {{
                            {fname}_functions[{idx}] = {type}_{fname}_{isa};
                        }}
                        #endif
                        """).format(
                            ISA=vt.upper(), isa=vt,
                            fname=name, type=tname, idx=k
                        ))
            else:
                funclist.append('NULL')
                try:
                    thedict = arity_lookup[uf.nin, uf.nout]
                except KeyError:
                    raise ValueError("Could not handle {}[{}]".format(name, t.type))

                astype = ''
                if not t.astype is None:
                    astype = '_As_%s' % thedict[t.astype]
                astr = ('%s_functions[%d] = PyUFunc_%s%s;' %
                           (name, k, thedict[t.type], astype))
                code2list.append(astr)
                if t.type == 'O':
                    astr = ('%s_data[%d] = (void *) %s;' %
                               (name, k, t.func_data))
                    code2list.append(astr)
                    datalist.append('(void *)NULL')
                elif t.type == 'P':
                    datalist.append('(void *)"%s"' % t.func_data)
                else:
                    astr = ('%s_data[%d] = (void *) %s;' %
                               (name, k, t.func_data))
                    code2list.append(astr)
                    datalist.append('(void *)NULL')
                    #datalist.append('(void *)%s' % t.func_data)
                sub += 1

            for x in t.in_ + t.out:
                siglist.append('NPY_%s' % (english_upper(chartoname[x]),))

            k += 1

        funcnames = ', '.join(funclist)
        signames = ', '.join(siglist)
        datanames = ', '.join(datalist)
        code1list.append("static PyUFuncGenericFunction %s_functions[] = {%s};"
                         % (name, funcnames))
        code1list.append("static void * %s_data[] = {%s};"
                         % (name, datanames))
        code1list.append("static char %s_signatures[] = {%s};"
                         % (name, signames))
    return "\n".join(code1list), "\n".join(code2list)

def make_ufuncs(funcdict):
    code3list = []
    names = sorted(funcdict.keys())
    for name in names:
        uf = funcdict[name]
        mlist = []
        docstring = textwrap.dedent(uf.docstring).strip()
        docstring = docstring.encode('unicode-escape').decode('ascii')
        docstring = docstring.replace(r'"', r'\"')
        docstring = docstring.replace(r"'", r"\'")
        # Split the docstring because some compilers (like MS) do not like big
        # string literal in C code. We split at endlines because textwrap.wrap
        # do not play well with \n
        docstring = '\\n\"\"'.join(docstring.split(r"\n"))
        if uf.signature is None:
            sig = "NULL"
        else:
            sig = '"{}"'.format(uf.signature)
        fmt = textwrap.dedent("""\
            identity = {identity_expr};
            if ({has_identity} && identity == NULL) {{
                return -1;
            }}
            f = PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
                {name}_functions, {name}_data, {name}_signatures, {nloops},
                {nin}, {nout}, {identity}, "{name}",
                "{doc}", 0, {sig}, identity
            );
            if ({has_identity}) {{
                Py_DECREF(identity);
            }}
            if (f == NULL) {{
                return -1;
            }}
        """)
        args = dict(
            name=name, nloops=len(uf.type_descriptions),
            nin=uf.nin, nout=uf.nout,
            has_identity='0' if uf.identity is None_ else '1',
            identity='PyUFunc_IdentityValue',
            identity_expr=uf.identity,
            doc=docstring,
            sig=sig,
        )

        # Only PyUFunc_None means don't reorder - we pass this using the old
        # argument
        if uf.identity is None_:
            args['identity'] = 'PyUFunc_None'
            args['identity_expr'] = 'NULL'

        mlist.append(fmt.format(**args))
        if uf.typereso is not None:
            mlist.append(
                r"((PyUFuncObject *)f)->type_resolver = &%s;" % uf.typereso)
        mlist.append(r"""PyDict_SetItemString(dictionary, "%s", f);""" % name)
        mlist.append(r"""Py_DECREF(f);""")
        code3list.append('\n'.join(mlist))
    return '\n'.join(code3list)


def make_code(funcdict, filename):
    code1, code2 = make_arrays(funcdict)
    code3 = make_ufuncs(funcdict)
    code2 = indent(code2, 4)
    code3 = indent(code3, 4)
    code = textwrap.dedent(r"""

    /** Warning this file is autogenerated!!!

        Please make changes to the code generator program (%s)
    **/
    #include "Python.h"
    #include "numpy/ufuncobject.h"
    #include "loops_intel.h"
    %s

    static int
    InitOperators(PyObject *dictionary) {
        PyObject *f, *identity;

    %s
    %s

        return 0;
    }
    """) % (filename, code1, code2, code3)
    return code


if __name__ == "__main__":
    filename = __file__
    code = make_code(defdict, filename)
    with open('__umath_generated.c', 'w') as fid:
        fid.write(code)
