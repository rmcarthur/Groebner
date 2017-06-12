from __future__ import division, print_function
import numpy as np
from scipy.signal import convolve, fftconvolve
from polynomial import Polynomial
import itertools

"""
1/11/17
Author: Rex McArthur
Creates a class of n-dim Power Basis polynomials. Tracks leading term,
coefficents, and inculdes basic operations (+,*,scaler multip, etc.)
Assumes GRevLex ordering, but should be extended.
Mostly used for testing vs other solvers
"""

class MultiPower(Polynomial):
    """
    _____ params _______
    dim: int, number of variables, dimension of polynomial system
    terms: int, highest term of single variable power polynomials
    coeff: list(terms**dim) or np.array ([terms,] * dim), coefficents in given ordering
    order: string, monomial ordering desired for Grobner calculations
    lead_term: list, the index of the current leading coefficent



    _____ methods ______
    next_step:
        input- Current: list, current location in ordering
        output- the next step in ordering
    """

    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        super(MultiPower, self).__init__(coeff, order, lead_term, clean_zeros)

    def __add__(self,other):
        '''
        Here we add an addition class.
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self,other)
        else:
            new_self, new_other = self, other
        return MultiPower(new_self.coeff + new_other.coeff)

    def __sub__(self,other):
        '''
        Here we subtract the two polys
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self,other)
        else:
            new_self, new_other = self, other
        return MultiPower(new_self.coeff - new_other.coeff)

    def __mul__(self,other):
        '''
        here we add leading terms?
        '''
        return MultiPower(fftconvolve(self.coeff, other.coeff))

    def match_size(self,a,b):
        '''
        Matches the shape of the polynomials
        '''
        new_shape = [max(i,j) for i,j in itertools.zip_longest(a.shape, b.shape, fillvalue = 0)] #finds the largest length in each dimmension
        # finds the difference between the largest length and the original shapes in each dimmension.
        add_a = [i-j for i,j in itertools.zip_longest(new_shape, a.shape, fillvalue = 0)]
        add_b = [i-j for i,j in itertools.zip_longest(new_shape, b.shape, fillvalue = 0)]
        #create 2 matrices with the number of rows equal to number of dimmensions and 2 columns
        add_a_list = np.zeros((len(new_shape),2))
        add_b_list = np.zeros((len(new_shape),2))
        #changes the second column to the values of add_a and add_b.
        add_a_list[:,1] = add_a
        add_b_list[:,1] = add_b
        #uses add_a_list and add_b_list to pad each polynomial appropriately.
        a = MultiPower(np.pad(a.coeff,add_a_list.astype(int),'constant'), clean_zeros = False)
        b = MultiPower(np.pad(b.coeff,add_b_list.astype(int),'constant'), clean_zeros = False)
        return a,b

    def __eq__(self,other):
        '''
        check if coeff matrix is the same
        '''
        if self.shape != other.shape:
            return False
        else:
            return np.allclose(self.coeff, other.coeff)

    def __ne__(self,other):
        '''
        check if coeff matrix is not the same same
        '''
        return not (self == other)

    def mon_mult(self,M):
        '''
        M is a tuple of the powers in the monomial.
            Ex: x^3*y^4*z^2 would be input as (3,4,2)
        #P is the polynomial.
        '''
        tuple1 = []
        for i in M:
            list1 = (i,0)
            tuple1.append(list1)
        return MultiPower(np.pad(self.coeff, tuple1, 'constant', constant_values = 0), clean_zeros = False)
