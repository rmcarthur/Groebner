from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve, convolve
import itertools
from polynomial import Polynomial

"""
08/31/17
Author: Rex McArthur
Creates a class of n-dim chebyshev polynomials. Tracks leading term,
coefficents, and inculdes basic operations (+,*,scalar multip, etc.)
Assumes GRevLex ordering, but should be extended.
"""


class MultiCheb(Polynomial):
    """
    _____ params _______
    dim: int, number of variables, dimension of chebyshev system
    terms: int, highest term of single variable chebyshev polynomials
    coeff: list(terms**dim) or np.array ([terms,] * dim), coefficents in given ordering
    order: string, monomial ordering desired for Groebner calculations
    lead_term: list, the index of the current leading coefficent



    _____ methods ______
    next_step:
        input- Current: list, current location in ordering
        output- the next step in ordering
    """

    def __init__(self, coeff, order='degrevlex', lead_term=None):
        super(MultiCheb, self).__init__(coeff, order, lead_term)


    def __add__(self,other):
        '''
        Here we add an addition method
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self,other)
        else:
            new_self, new_other = self, other

        return MultiCheb(new_self.coeff + new_other.coeff)

    def __sub__(self,other):
        '''
        Here we subtract the two polys coeffs
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self,other)
        else:
            new_self, new_other = self, other
        return MultiCheb(new_self.coeff - new_other.coeff)


    def match_size(self,a,b):
        '''
        Matches the size of the polynomials
        '''
        new_shape = [max(i,j) for i,j in itertools.zip_longest(a.shape, b.shape)]
        add_a = [i-j for i,j in zip(new_shape, a.shape)]
        add_b = [i-j for i,j in zip(new_shape, b.shape)]
        add_a_list = np.zeros((2,len(new_shape)))
        add_b_list = np.zeros((2,len(new_shape)))
        add_a_list[:,1] = add_a
        add_b_list[:,1] = add_b
        a = MultiCheb(np.pad(a.coeff,add_a_list.astype(int),'constant'))
        b = MultiCheb(np.pad(b.coeff,add_b_list.astype(int),'constant'))
        return a,b

    def __mul__(self,other):
        '''
        Multiply by convolving intelligently
        CURRENTLY ONLY DOING 2-D support
        Manually make 1, 3D support then add n-dim support
        '''
        # Check and see if same size
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self,other)
        else:
            new_self, new_other = self, other
        c = new_other.coeff[::-1, ::-1]
        p1 = convolve(new_self.coeff,new_other.coeff)
        temp = convolve(new_self.coeff,c)
        half = len(p1)//2
        p2 = temp[:half+1,:][::-1] + temp[half:,:]
        p2[0,:] = p2[0,:]/2.
        p2 = p2[:,:half+1][:, ::-1] + p2[:,half:]
        p2[:,0] = p2[:,0]/2.
        p_z = np.zeros_like(p1)
        p_z[:half+1, :half+1] = p2
        new_coeff = .5*(p1 + p_z)
        #TODO: You can use the lead_term kwarg to save some time
        return MultiCheb(new_coeff)
