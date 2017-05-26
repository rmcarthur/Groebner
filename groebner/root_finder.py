'''
This class represents the ring C[x1,...,xn]/I as a vector space over C.
'''
import numpy as np
from groebner_class import Groebner
import itertools

class RootFinder(object):
    '''
    attributes
    ----------
    self.GroebnerBasis : Groebner object
        provides methods for calculating a Groebner basis and for
        dividing a polynomial by a set of other polynomials
    self.GB : list
        polynomials in Groebner basis
    self.vectorBasis : list
        tuples representing monomials in the vector space basis
    self.dimension : int
        dimension of the vector space
    '''
    def __init__(self, Groebner):
        '''
        parameters
        ----------
        Groebner : Groebner object or list
            groebner object that represents a Groebner Basis for the ideal
            OR
            a list of polynomials that make up a Groebner basis
        '''
        if type(Groebner) is list:
            self.Groebner = Groebner()
            self.GB = Groebner
        else:
            self.Groebner = Groebner
            self.GB = Groebner.solve()

        self.vectorBasis = self.makeVectorBasis(self.GB)
        self.vectorSpaceDimension = len(self.vectorBasis)

    def makeVectorBasis(self, GB):
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

    def multOperatorMatrix(self, poly):


    def coordinateVector(self, reducedPoly):
        '''
        parameters
        ----------
        reducedPoly : polynomial object
            the polynomial for which to find the coordinate vector of its coset

        return
        ------
        coordinateVector : list
            the coordinate vector of the given polynomial's coset in
            A = C[x_1,...x_n]/I as a vector space over C
        '''
        # reverse the array since self.vectorBasis is in increasing order
        # and monomialList() gives a list in decreasing order
        reducedPolyTerms = reducedPoly.monomialList()[::-1]
        assert(len(reducedPolyTerms) <= self.vectorSpaceDimension)

        coordinateVector = [0] * self.vectorSpaceDimension
        for monomial in reducedPolyTerms:
            coordinateVector[self.vectorBasis.index(monomial)] = \
                reducedPoly.coeff[monomial]

        return coordinateVector

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
        return all(np.subtract(mon2, mon1) >= 0)

    def getRemainder(self, poly):
        '''
        parameters
        ----------
        polynomial : polynomial object, either power or chebychev
            the polynomial to be divided by the Groebner basis

        return
        ------
        polynomial object
            the unique remainder of poly divided by self.GB
        '''
        return self.Groebner.reduce_poly(polynomial, self.GB)
