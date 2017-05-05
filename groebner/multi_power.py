from __future__ import division, print_function
import numpy as np
from scipy.signal import convolve, fftconvolve

"""
1/11/17
Author: Rex McArthur
Creates a class of n-dim Power Basis polynomials. Tracks leading term,
coefficents, and inculdes basic operations (+,*,scaler multip, etc.)
Assumes GRevLex ordering, but should be extended.
Mostly used for testing vs other solvers
""" 

class MultiPower(object):
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

    def __init__(self, coeff, order='grevlex', lead_term=None):
        '''
        terms, int- number of chebyshev polynomials each variable can have. Each dimension will have term terms
        dim, int- number of different variables, how many dim our tensor will be
        order, string- how you want to order your polynomials. Grevlex is default
        '''
        self.coeff = coeff
        self.dim = self.coeff.ndim
        self.terms = np.prod(self.coeff.shape)
        self.order = order
        self.shape = self.coeff.shape
        self.max_term = np.max(self.shape) -1

        if lead_term is None:
            self.update_lead_term()
        else:
            self.lead_term = lead_term


    def check_column_overload(self, max_values, current, column):
        '''
        Checks to make sure that we aren't going into the negatives, aka the current value can't ever be greater
        than the max_values value. We check at the column where we have just added stuff and might have an 
        overflow
        Return true if the whole thing is full and needs to increment i again. False otherwise.
        '''
        initial_column = column
        if(current[column] > max_values[column]):
            initial_amount = current[column]
            extra = current[column] - max_values[column]
            current[column] = max_values[column]
            while(extra>0):
                if(column==0):
                    current[0] += extra
                    #Sets all the stuff back in the initial row, needed if the while loop is used.
                    for i in range(0, initial_column):
                        current[i+1] += current[i]
                        current[i] = 0
                    return True
                else:
                    column -=1
                    allowed = max_values[column] - current[column]
                    if(allowed > extra):
                        current[column] += extra
                        extra = 0
                    else:
                        current[column] += allowed
                        extra -= allowed
            return False
        else:
            return False

    def degrevlex_gen(self):
        '''
        yields grevlex ordering co-ordinates in order to find
        the leading coefficent
        '''
        max_values = tuple(self.shape)-np.ones_like(self.shape)
        base = max_values
        current = np.zeros(self.dim)
        yield base-current
        while True:
            for i in range(1, sum(max_values)+1):
                onward = True
                #set the far right column to i
                current = np.zeros(self.dim)
                current[self.dim-1] = i
                #This can't return false, as we start at the begenning. Always has enough room to spill over.
                self.check_column_overload(max_values, current, self.dim-1)
                yield base - current
                while onward:
                    #Find the leftmost thing
                    for j in range(0, self.dim):
                        if(current[j] != 0):
                            left_most_spot = j
                            break
                    if(left_most_spot != 0):
                        #Slide it to the left
                        current[left_most_spot] -= 1
                        current[left_most_spot-1] += 1
                        yield base - current
                    elif(current[j] == i):
                        #Reset it for the next run
                        current[0] = 0
                        onward = False
                    else:
                        #if I'm at the end push back everything to the next leftmost thing and slide it plus 1
                        amount = current[0]
                        for j in range(1,self.dim):
                            if(current[j] != 0):
                                next_left_most_spot = j
                                break
                        current[0] = 0
                        current[next_left_most_spot] -= 1
                        current[next_left_most_spot-1] += amount+1

                        spot_to_check = next_left_most_spot-1
                        #Loops throught this until everything is balanced all right or we need to increase i
                        while(self.check_column_overload(max_values, current, spot_to_check)):
                            new_spot_to_check = -1
                            for j in range(spot_to_check+1, self.dim):
                                if(current[j] != 0):
                                    new_spot_to_check = j
                                    break
                            if(new_spot_to_check == -1):
                                onward = False
                                break
                            else:
                                amount = current[spot_to_check]
                                current[spot_to_check] = 0
                                current[new_spot_to_check] -=1
                                current[new_spot_to_check-1] += (amount+1)
                                spot_to_check = new_spot_to_check-1
                        if(onward):
                            yield base-current
            return

    def update_lead_term(self,start = None):
        #print('Updating Leading Coeff...')
        if self.order == 'grevlex':
            gen = self.degrevlex_gen()
            for idx in gen:
                if self.coeff[tuple(idx)] != 0:
                    self.lead_term = idx
                    self.lead_coeff = self.coeff[tuple(idx)]
                    break
        #print('Leading Coeff is {}'.format(self.lead_term))


    def __lt__(self, other):
        '''
        Magic method for determing which polynomial is smaller
        #TODO: Fix so this works for things of different lengths
        '''
        if sum(self.lead_term) < sum(other.lead_term):
            return True

        elif sum(self.lead_term) > sum(other.lead_term):
            return False
        
        else:
            for i in xrange(len(self.lead_term)):
                if self.lead_term[i] < other.lead_term[i]: 
                    return True
                if self.lead_term[i] > other.lead_term[i]:
                    return False
            if self.coeff[tuple(self.lead_term)] < other.coeff[tuple(other.lead_term)]:
                return True


    def __gt__(self, other):
        '''
        Magic method for determing which polynomial is smaller
        #TODO: Fix so this works for things of different lengths
        '''
        if sum(self.lead_term) < sum(other.lead_term):
            return False

        elif sum(self.lead_term) > sum(other.lead_term):
            return True
        
        else:
            for i in xrange(len(self.lead_term)):
                if self.lead_term[i] < other.lead_term[i]: 
                    return False
                if self.lead_term[i] > other.lead_term[i]:
                    return True
            if self.coeff[tuple(self.lead_term)] < other.coeff[tuple(other.lead_term)]:
                return False

    def __add__(self,other):
        '''
        Here we add an addition class.
        '''
        return MultiPower(self.coeff + other.coeff)

    def __sub__(self,other):
        '''
        Here we subtract the two polys
        '''
        return MultiPower(self.coeff - other.coeff)

    def __mul__(self,other):
        '''
        here we add leading terms?
        '''
        return MultiPower(fftconvolve(self.coeff, other.coeff))

