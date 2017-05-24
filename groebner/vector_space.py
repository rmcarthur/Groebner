'''
This class represents the ring C[x1,...,xn]/I as a vector space over C.
'''
import numpy as np
from groebner_class import Groebner
import itertools

class VectorSpace(object):
    '''
    attributes
    ----------
    self.GB : list
        polynomials in Groebner basis
    self.basis : list
        tuples representing monomials in the vector space basis
    self.dimension : int
        dimension of the vector space
    '''
    def __init__(self, GroebnerBasis):
        '''
        parameters
        ----------
        GroebnerBasis : list
            polynomial objects that make up a Groebner Basis for the
            ideal of interest
        '''
        self.GB = GroebnerBasis
        self.basis = self.makeBasis(self.GB)
        self.dimension = len(self.basis)

    def makeBasis(self, GB):
        '''
        parameters
        ----------
        GB: list
            polynomial objects that make up a Groebner basis for the ideal

        return
        ------
        basis : list
            tuples representing the monomials in the vector space basis
        '''
        LT_G = [f.lead_term for f in GB]
        possibleVarDegrees = [range(max(tup)) for tup in zip(*LT_G)]
        possibleMonomials = itertools.product(*possibleVarDegrees)
        basis = []

        for mon in possibleMonomials:
            divisible = False
            for LT in LT_G:
                if (self.divides(LT, mon)):
                     divisible = True
                     break
            if (not divisible):
                basis.append(mon)

        return basis

    def divides(self, mon1, mon2):
        '''
        parameters
        ----------
        mon1 : tuple
            contains the exponents of the monomial divisor
        mon2 : tuple
            contains the exponents of the monomial dividend

        return
        ------
        boolean
            true if mon1 divides mon2, false otherwise
        '''
        return not any(np.subtract(mon2, mon1) < 0)

    
