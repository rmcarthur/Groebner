import numpy as np
from polynomial import Polynomial
from numpy.polynomial import chebyshev as C
from multi_cheb import MultiCheb
from multi_power import MultiPower

def conv_cheb(T):
    conv = C.cheb2poly(T)
    if conv.size == T.size:
        return conv
    else:
        pad = T.size - conv.size
        new_conv = np.pad(conv, ((0,pad)), 'constant')
        return new_conv

def conv_poly(P):
    conv = C.poly2cheb(P)
    if conv.size == P.size:
        return conv
    else:
        pad = P.size - conv.size
        new_conv = np.pad(conv, ((0,pad)), 'constant')
        return new_conv

def cheb2poly(T):
    dim = len(T.shape)
    A = T.coeff
    for i in range(dim):
        A = np.apply_along_axis(conv_cheb, i, A)
    return MultiPower(A)

def poly2cheb(P):
    dim = len(P.shape)
    A = P.coeff
    for i in range(dim):
        A = np.apply_along_axis(conv_poly, i, A)
    return MultiCheb(A)
