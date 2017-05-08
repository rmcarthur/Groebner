from __future__ import print_function, division
import fractions
import itertools
import numpy as np
import pandas as pd
import maxheap
import os,sys
from multi_cheb import MultiCheb
from multi_power import MultiPower
from scipy.linalg import lu

class Groebner(object):

    def __init__(self,polys):
        '''
        polys -- a list of polynomials that generate your ideal
        self.org_len - Orginal length of the polys passed in
        '''
        self.polys = polys
        self.f_len = len(polys)
        self.largest_mon = maxheap.TermOrder(tuple((0,0)))
        self.matrix = pd.DataFrame()
        self.label = [] # want to drop this
        self.label_count = 0 # and this
        self.np_matrix = np.zeros([0,0]) # and this
        self.term_set = set()
        self.pd_term_set = set()
        self.term_dict = {}


        #np objects
        self.matrix_terms = [] #Instantiate  here?
        self.np_matrix = np.array([[]])

        self._add_polys(polys)

    def solve(self):
        while True:

            self._build_matrix()
            self.add_s_to_matrix()
	    self.matrix = self.matrix.loc[:, (self.matrix != 0).any(axis=0)]

            self.add_r_to_matrix()
	    self.matrix = self.matrix.loc[:, (self.matrix != 0).any(axis=0)]


            # Flip due to bad ordering in grevlex generator

            # Put correct order on table

            # Put R on top for reduction

            #P,L,U = lu(new_mat)
            #P_argmax = np.argmax(P,axis=0) 


    def _add_poly_to_matrix(self,p):
        '''
        Takes in a single polynomial and adds it to the state matrix
        '''
        for idx in p.grevlex_gen(): 
            idx_term = maxheap.Term(tuple(idx)) #Get a term object 
            # Grab each non-zero element, put it into matrix. 
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


        # THIS MAKES THE DF NEEDS TO BE REMOVED
        for poly in self.polys:
            #For each polynomial, make a matrix object, and add its column
            submatrix = pd.DataFrame()
            for idx in poly.grevlex_gen():
                idx_term = maxheap.TermOrder(tuple(idx)) # Used to get an ordering on terms
                if not idx_term.val in self.pd_term_set:
                    self.pd_term_set.add(idx_term.val)
                    self.label.append(tuple(idx)) # Put the actual tuple of index into a list
                submatrix[str(idx)] = pd.Series([poly.coeff[tuple(idx)]])
            #Append all submatracies
            self.matrix = self.matrix.append(submatrix)
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

        if isinstance(a, MultiPower) and isinstance(b,MultiPower):
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
        This is dead wrong, needs to check if they are lcm
        '''
        return False
    
    def add_s_to_matrix(self):
        '''
        This takes all possible combinaions of s polynomials and adds them to the Grobner Matrix
        '''
        for a, b in itertools.combinations(self.polys, 2):
            submatrix = pd.DataFrame()
            #if not self._coprime(a.lead_coeff,b.lead_coeff): #Checks for co-prime coeffs
            s = self.calc_s(a,b) # Calculate the S polynomail

            for idx in s.grevlex_gen():
                idx_term = maxheap.TermOrder(tuple(idx)) # For each term in polynomial, throw it on the heap
                if not idx_term.val in self.term_set: # Add all new polynomials
                    self.term_set.add(idx_term.val)
                    self.label.append(tuple(idx))
                    if idx_term > self.largest_mon:
                        self.largest_mon = idx_term
                submatrix[str(idx)] = pd.Series([s.coeff[tuple(idx)]]) 
            self.matrix = self.matrix.append(submatrix)
            self.matrix = self.matrix.fillna(0)
            self.fs_len = len(self.matrix.index)
            pass

    def add_poly_to_matrix(self,p):
        submatrix = pd.DataFrame()
        for idx in p.grevlex_gen():
            submatrix[str(idx)] = pd.Series([p.coeff[tuple(idx)]])
        self.matrix = self.matrix.append(submatrix)
        self.matrix = self.matrix.fillna(0)
        pass

    def add_r_to_matrix(self):
        '''
        Makes Heap out of all monomials, and finds lcms to add them into the matrix
        '''
        for monomial in self.term_set:
            m = list(monomial)
            for p in self.polys:
                l = list(p.lead_term)
                if all([i<=j for i,j in zip(l,m)]) and len(l) == len(m):
                    c = [j-i for i,j in zip(l,m)]
                    c_coeff = np.zeros(np.array(self.largest_mon.val)+1)
                    c_coeff[tuple(c)] = 1 
                    if isinstance(p, MultiCheb):
                        c = MultiCheb(c_coeff)
                    elif isinstance(p,MultiPower):
                        c = MultiPower(c_coeff)
                    r = c*p
                    self.add_poly_to_matrix(r)
                    break
        pass 



