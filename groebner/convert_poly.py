import numpy as np
from scipy.signal import fftconvolve, convolve
import itertools
from polynomial import Polynomial
from numpy.polynomial import chebyshev as C
from multi_cheb import MultiCheb
from multi_power import MultiPower

def cheb_poly(T): #T is a MultiCheb object, function will return a MultiPower object
    '''
    Assume that T is a 2D object for now. Extend dimensions later

    Idea:
        1. slice along each row and convert 1D rows to polynomials
        2. Each row will be multiplied by corresponding chebyshev
    '''
    y_poly = []
    for i in range(50):
        cheby = np.zeros(i+1)
        cheby[i] = 1
        poly = C.cheb2poly(cheby)
        poshape = poly.shape
        poly = poly.reshape(poshape[0],1)
        y_poly.append(MultiPower(poly))
    A = T.coeff
    shape = A.shape
    solution = MultiPower(np.zeros_like(A))
    for i in range(shape[0]):
        new_poly = MultiPower(C.cheb2poly(A[i,:]))
        temp = new_poly*y_poly[i]
        solution = solution + temp
    return solution

def poly_cheb(P): #P is a MultiPower object
    y_cheb = []
    for i in range(50):
        poly = np.zeros(i+1)
        poly[i] = 1
        cheb = C.poly2cheb(poly)
        chebshape = cheb.shape
        cheb = cheb.reshape(chebshape[0],1)
        y_cheb.append(MultiCheb(cheb))
    A = P.coeff
    shape = A.shape
    solution = MultiCheb(np.zeros_like(A))
    for i in range(shape[0]):
        new_cheb = MultiCheb(C.poly2cheb(A[i,:]))
        temp = new_cheb*y_cheb[i]
        solution = solution + temp
    return solution #returns a MultiCheb object
