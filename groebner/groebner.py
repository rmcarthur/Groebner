from __future__ import print_function, division
import fractions
import itertools
import numpy as np
import maxheap
import os,sys
from multi_cheb import MultiCheb
from multi_power import MultiPower
from scipy.linalg import lu
from numpy.linalg import qr

class Groebner(object):

    def __init__(self,polys):
        '''
        polys -- a list of polynomials that generate your ideal
        self.org_len - Orginal length of the polys passed in
        '''
        self.old_polys = list()
        self.new_polys = polys
        self.f_len = len(polys)
        self.largest_mon = maxheap.TermOrder(tuple((0,0)))
        self.label = [] # want to drop this
        self.label_count = 0 # and this
        self.np_matrix = np.zeros([0,0]) # and this
        self.term_set = set()
        self.term_dict = {}

        #np objects
        self.matrix_terms = [] #Instantiate  here?
        self.np_matrix = np.array([[]])
        self._add_polys(polys)
    
    def solve(self):
        polys_added = True
        while polys_added:

            self._build_matrix()
            self.add_s_to_matrix()
            self.matrix = self.matrix.loc[:, (self.matrix != 0).any(axis=0)]

            self.add_r_to_matrix()
            self.matrix = self.matrix.loc[:, (self.matrix != 0).any(axis=0)]

            polys_added = self.reduce(self)
            # Flip due to bad ordering in grevlex generator

            # Put correct order on table

            # Put R on top for reduction

            #P,L,U = lu(new_mat)
            #P_argmax = np.argmax(P,axis=0) 
        print(self.old_polys)

    def _add_poly_to_matrix(self,p):
        '''
        Takes in a single polynomial and adds it to the state matrix
        '''
        for idx in p.degrevlex_gen(): 
            idx_term = maxheap.Term(tuple(idx)) #Get a term object 
            # Grab each non-zero element, put it into matrix. 
            idx_term.val = tuple(map(lambda i: int(i), idx_term.val))
            coeff_val = p.coeff[idx_term.val] 

            # If already in idx_list
            if idx_term.val in self.term_set:
                # get index of label and np matrix to put into
                idx_where = np.argmax([i == idx_term for i in self.matrix_terms]) 
                self.np_matrix[0,idx_where] = coeff_val

            # If new column needed
            else:
                # Make new column
                self.term_set.add(idx_term.val)
                length_of_mat = self.np_matrix.shape[0]
                if length_of_mat == 0:
                    self.np_matrix = np.zeros((1,1))
                else:
                    zeros = np.zeros((length_of_mat,1))
                    self.np_matrix = np.hstack((self.np_matrix, zeros))
                self.matrix_terms.append(idx_term)
                self.np_matrix[0,-1] = coeff_val
        zero_poly = np.zeros((1,self.np_matrix.shape[1]))
        self.np_matrix = np.vstack((zero_poly,self.np_matrix))
        pass

    def _add_polys(self, p_list):
        '''
        Adds a single polynomial to the state matrix
        If an index doesn't exist yet, it adds a new column of zeros, to be sorted at the end
        params:
        p - a single polynomial object

        sets self.matrix to the appropriate matrix
        
        returns - None
        '''
        self.np_matrix = np.array([[]])
        for p in p_list:
            # Add a zero row for this polynomial
            self._add_poly_to_matrix(p)

        self.np_matrix = self.np_matrix[1:,:]

        argsort_list, self.matrix_terms = self.argsort(self.matrix_terms)
        self.np_matrix = self.np_matrix[:, argsort_list]
        pass 

    def argsort(self, index_list):
        '''
        Returns an argsort list for the index, as well as sorts the list in place
        '''
        argsort_list = sorted(range(len(index_list)), key=index_list.__getitem__)[::-1]
        index_list.sort()
        return argsort_list, index_list[::-1]
    

    def _lcm(self,a,b):
        '''
        Finds the LCM of the two leading terms of Polynomial a,b
        
        Params:
        a, b - polynomail objects
    
        returns:
        LCM - the np.array of the lead_term of the lcm polynomial
        '''
        return np.maximum(a.lead_term, b.lead_term)        
    
    def calc_s(self,a,b):
        '''
        Calculates the S-polynomial of a,b
        '''
        lcm = self._lcm(a,b)
        a_coeffs = np.zeros_like(a.coeff)
        a_coeffs[tuple([i-j for i,j in zip(lcm, a.lead_term)])] = 1.

        b_coeffs = np.zeros_like(b.coeff)
        b_coeffs[tuple([i-j for i,j in zip(lcm, b.lead_term)])] = 1.

        print(type(MultiPower))
        print(type(a))
        #print(isinstance(a, MultiPower), isinstance(b, MultiPower))
        print(isinstance(a,MultiPower))
        print(isinstance(a,MultiCheb))
        print(type(a) == type(b))
        if isinstance(a, MultiPower) and isinstance(b, MultiPower):
            b_ = MultiPower(b_coeffs)
            a_ = MultiPower(a_coeffs)
        elif isinstance(a, MultiCheb) and isinstance(b,MultiCheb):
            b_ = MultiCheb(b_coeffs)
            a_ = MultiCheb(a_coeffs)
        else:
            raise ValueError('Incompatiable polynomials')
        
        s = a_ * a - b_ * b
        return s

    def _coprime(self,a,b):
        '''
        needs to check if they are lcm
        '''
        return False
    
    def add_s_to_matrix(self):
        '''
        This takes all possible combinaions of s polynomials and adds them to the Grobner Matrix
        '''
        for a, b in itertools.combinations(self.polys, 2):
            #if not self._coprime(a.lead_coeff,b.lead_coeff): #Checks for co-prime coeffs

            # Calculate the S polynomail
            s = self.calc_s(a,b) 

            # Add it to np_matrix
            self._add_poly_to_matrix(s)

        argsort_list, self.matrix_terms = self.argsort(self.matrix_terms)
        self.np_matrix = self.np_matrix[:, argsort_list]
        pass



    def _build_maxheap(self):
        '''
        Builds a maxheap for use in r polynomial calculation
        '''
        self.monheap = maxheap.MaxHeap()
        for mon in self.term_set:
            self.monheap.heappush(mon)
        pass

    def add_r_to_matrix(self):
        '''
        Makes Heap out of all monomials, and finds lcms to add them into the matrix
        '''
        self._build_maxheap()
        while len(self.monheap) > 0:
            m = list(self.monheap.heappop())
            #for monomial in self.term_set:
            #m = list(monomial)
            #print("M: ", m)
            for p in self.polys:
                l = list(p.lead_term)
                #print("L: ",l)
                if all([i<=j for i,j in zip(l,m)]) and len(l) == len(m):
                    c = [j-i for i,j in zip(l,m)]
                    #print(c)
                    c_coeff = np.zeros(np.array(self.matrix_terms[0].val)+1)
                    c_coeff[tuple(c)] = 1 
                    if isinstance(p, MultiCheb):
                        c = MultiCheb(c_coeff)
                    elif isinstance(p,MultiPower):
                        c = MultiPower(c_coeff)
                    r = c*p
                    self._add_poly_to_matrix(r)
                else:
                    #print('Bad i,j or length')
                    break

        # Resort according to monomial ordering
        argsort_list, self.matrix_terms = self.argsort(self.matrix_terms)
        self.np_matrix = self.np_matrix[:, argsort_list]

        # Remove all zero polynomials, and monomials with no elements in them
        non_zero_mon = abs(np.sum(self.np_matrix, axis=0))!=0
        self.np_matrix = self.np_matrix[:,non_zero_mon]
        non_zero_poly = abs(np.sum(self.np_matrix,axis=1))!=0
        self.np_matrix = self.np_matrix[non_zero_poly,:]
        pass 
    
    def reduce_matrix(self, qr_decomposition=True):
        print(self.np_matrix)
        di={}
        for i, j in zip(*np.where(self.np_matrix!=0)):
            if i in di:
                continue
            else:
                di[i]=j
        old_lms = set(di.values())
        
        if qr_decomposition:
            Q,R = qr(self.np_matrix)
            reduced_matrix = R
        else:
            P,L,U = lu(self.np_matrix)
            reduced_matrix = U
        
        print(reduced_matrix)
        
        good_poly_spots = list()
        already_looked_at = set()
        for i, j in zip(*np.where(reduced_matrix!=0)):
            if i in already_looked_at:
                continue
            elif j in old_lms:
                already_looked_at.add(i)
                continue
            else:
                already_looked_at.add(i)
                good_poly_spots.append(i)
        
        print(good_poly_spots)
        
        self.old_polys = self.new_polys + self.old_polys
        self.new_polys = list()
        if(len(good_poly_spots) ==0):
            return False
        else:
            #self.new_polys = self.sm_to_poly(self, good_poly_spots)
            return True
