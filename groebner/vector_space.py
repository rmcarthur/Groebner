'''
This class represents the ring C[x1,...,xn]/I as a vector space over C.
'''
import numpy as np
from groebner_class import Groebner
import itertools

class VectorSpace(object):
    '''
    params:
    GB - a Groebner object representing a Groebner Basis
                     for the ideal of interest
    '''
    def __init__(self):
        '''
        attributes:
        self.GB - list of polynomials in Groebner basis
        self.basis - list of tuples representing monomials in the vector space basis
        '''
        self.GB = []
        self.basis = []

    def makeBasis(self, GB):
        '''
        param GB: a list of polynomial objects that make up a Groebner basis
                     for the ideal of interest

        return: a list of tuples representing the monomials in the vector space basis
        '''
        LT_G = [f.lead_term for f in GB]
        possibleVarDegrees = [range(max(tup)) for tup in zip(*LT_G)]
        possibleMonomials = itertools.product(*possibleVarDegrees)
        basis = []

        for mon in possibleMonomials:
            divisible = False
            for LT in LT_G:
                if (not any(np.subtract(mon, LT) < 0)):
                     divisible = True
                     break
            if (not divisible):
                basis.append(mon)

        return basis
