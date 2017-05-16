from operator import itemgetter
import itertools
import numpy as np
import maxheap
import os,sys
import math
from multi_cheb import MultiCheb
from multi_power import MultiPower
from scipy.linalg import lu
from scipy.linalg import qr

class Groebner(object):

    def __init__(self,polys):
        '''
        polys -- a list of polynomials that generate your ideal
        self.org_len - Orginal length of the polys passed in
        '''
        self.old_polys = list()
        self.new_polys = polys
        self.f_len = len(polys)
        self.largest_mon = maxheap.Term(tuple((0,0)))
        self.np_matrix = np.zeros([0,0]) # and this
        self.term_set = set()
        self.lead_term_set = set()
        self.term_dict = {}

        # Check polynomial types

        #print([type(p) == groebner.multi_power.MultiPower for p in self.new_polys])
        if all([type(p) == MultiPower for p in self.new_polys]):
            self.power = True
        elif all([type(p) == MultiCheb for p in self.new_polys]):
            self.power = False
        else:
            print([type(p) == MultiPower for p in self.new_polys])
            raise ValueError('Bad polynomials in list')

        #np objects
        #initialize_np_matrix
    
    def initialize_np_matrix(self):
        self.matrix_terms = []
        self.np_matrix = np.array([[]])
        self.term_set = set()
        self.lead_term_set = set()
        self._add_polys(self.new_polys)
        self._add_polys(self.old_polys)
        #self.sort_matrix()
        pass
    
    def solve(self):
        polys_were_added = True
        while polys_were_added:
            print("Starting Loop")
            self.initialize_np_matrix()
            print(self.np_matrix.shape)
            print("ADDING PHI's")
            self.add_phi_to_matrix()
            print(self.np_matrix.shape)
            print("ADDING r's")
            self.add_r_to_matrix()
            print(self.np_matrix.shape)
            polys_were_added = self.reduce_matrix()
        print("WE WIN")
        for poly in self.old_polys:
            print(poly.coeff)
        pass
    
    def sort_matrix(self):
        # Sorts the matrix. 
        argsort_list, self.matrix_terms = self.argsort(self.matrix_terms)
        self.np_matrix = self.np_matrix[:,argsort_list]
        
        # Remove all zero polynomials, and monomials with no elements in them
        non_zero_mon = abs(np.sum(self.np_matrix, axis=0))!=0
        self.np_matrix = self.np_matrix[:,non_zero_mon]
        to_remove = set()
        for i,j in zip(self.matrix_terms, non_zero_mon):
            if not j:
                to_remove.add(i)
        for i in to_remove:
            self.matrix_terms.remove(i)
        non_zero_poly = abs(np.sum(self.np_matrix,axis=1))!=0
        self.np_matrix = self.np_matrix[non_zero_poly,:]
        pass

    def sm_to_poly(self,idxs,reduced_matrix):
        '''
        Takes a list of indicies corresponding to the rows of the state matrix and 
        returns a list of polynomial objects
        '''
        shape = []
        p_list = []
        matrix_term_vals = [i.val for i in self.matrix_terms]

        # Finds the maximum size needed for each of the poly coeff tensors
        for i in range(len(matrix_term_vals[0])):
            # add 1 to each to compensate for constant term
            shape.append(max(matrix_term_vals, key=itemgetter(i))[i]+1)

        # Grabs each polynomial, makes coeff matrix and constructs object
        for i in idxs:
            p = reduced_matrix[i]
            coeff = np.zeros(shape)
            for j,term in enumerate(matrix_term_vals):
                coeff[term] = p[j]            
            
            #Might be good to check for unneeded 0's here. No rush though.
            
            if self.power:
                poly = MultiPower(coeff)
            else:
                poly = MultiCheb(coeff)
            p_list.append(poly)
        return p_list

    def _add_poly_to_matrix(self,p):
        '''
        Takes in a single polynomial and adds it to the state matrix
        '''
        self.lead_term_set.add(p.lead_term)
        
        for idx in p.degrevlex_gen(): 
            idx_term = maxheap.Term(tuple(idx)) #Get a term object 
            # Grab each non-zero element, put it into matrix. 
            idx_term.val = tuple(map(lambda i: int(i), idx_term.val))
            coeff_val = p.coeff[idx_term.val] 

            # If already in idx_list
            if idx_term in self.term_set:
                # get index of label and np matrix to put into
                idx_where = np.argmax([i == idx_term for i in self.matrix_terms]) 
                self.np_matrix[0,idx_where] = coeff_val

            # If new column needed
            else:
                # Make new column
                self.term_set.add(idx_term)
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
        p_list - a list of polynomial object

        sets self.matrix to the appropriate matrix
        
        returns - None
        '''
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
    
    def calc_phi(self,a,b):
        '''Calculates the phi-polynomial's of the polynomials a and b.
        Returns:
            A tuple of the calculated phi's. 
        '''
        lcm = self._lcm(a,b)
        #updated to use monomial multiplication. Will crash for MultiCheb until that gets added
        a_diff = tuple([i-j for i,j in zip(lcm, a.lead_term)])
        b_diff = tuple([i-j for i,j in zip(lcm, b.lead_term)])
        return a.mon_mult(a_diff), b.mon_mult(b_diff)
    
    def add_phi_to_matrix(self):
        '''
        Takes all new possible combinations of phi polynomials and adds them to the Groebner Matrix
        
        '''
        for i,j in itertools.combinations(self.new_polys+self.old_polys,2):
            # This prevents calculation of phi with combinations of old_f exclusively. 
            if i not in self.old_polys:
                # Relative prime check: If the elementwise multiplication of list i and j are all zeros, calculation of phi is not needed. 
                #(I separated the if statements for better visibility reasons, if it's better to combine, please fix!)
                if not all([ a == 0 or b ==0 for a,b in zip(np.ndarray.flatten(i.coeff), np.ndarray.flatten(j.coeff))]): 
                # Calculate the phi's.
                    p_a , p_b = self.calc_phi(i,j)
                    # Add the phi's on to the Groebner Matrix. 
                    self._add_poly_to_matrix(p_a)
                    self._add_poly_to_matrix(p_b)
        #self.sort_matrix()
        pass
    
    def _build_maxheap(self):
        '''
        Builds a maxheap for use in r polynomial calculation
        '''
        self.monheap = maxheap.MaxHeap()        
        for mon in self.term_set:
            if(mon.val not in self.lead_term_set):
                self.monheap.heappush(mon)
        pass

    def calc_r(self, m):
        '''
        Finds the r polynomial that has a leading monomial m
        Returns the polynomial.
        '''
        for p in self.new_polys + self.old_polys: #Do we need all of these?
                l = list(p.lead_term)
                if all([i<=j for i,j in zip(l,m)]) and len(l) == len(m):
                    #New mon_mult method. Will break with MultiCheb unitl that is added.
                    c = [j-i for i,j in zip(l,m)]
                    if not l == m: #make sure c isn't all 0
                        return p.mon_mult(c)
        return None

    def add_r_to_matrix(self):
        '''
        Makes Heap out of all monomials, and finds lcms to add them into the matrix
        '''
        self._build_maxheap()
        while len(self.monheap) > 0:
            m = list(self.monheap.heappop().val)
            r = self.calc_r(m)
            if not r==None:
                self._add_poly_to_matrix(r)
        self.sort_matrix()
        pass
    
    def reduce_matrix(self, qr_decomposition=False):
        di={}
        for i, j in zip(*np.where(self.np_matrix!=0)):
            if i in di:
                continue
            else:
                di[i]=j
        old_lms = set(di.values())
                
        if qr_decomposition:
            #print(self.np_matrix)
            Q,R = qr(self.np_matrix, 'pivoting' == True)
            reduced_matrix = R
            #print(R)
        else:
            P,L,U = lu(self.np_matrix)
            reduced_matrix = U
        
        #Check that it's fully reduced
        
        good_poly_spots = list()
        already_looked_at = set()
        for i, j in zip(*np.where(reduced_matrix!=0)):
            if i in already_looked_at:
                continue
            elif j in old_lms:
                already_looked_at.add(i)
                continue
            else:
                #old_lms.add(j) #until we get better reducing
                already_looked_at.add(i)
                good_poly_spots.append(i)
        self.old_polys = self.new_polys + self.old_polys
        self.new_polys = list()
        if(len(good_poly_spots) ==0):
            return False
        else:
            self.new_polys = self.sm_to_poly(good_poly_spots, reduced_matrix)
            return True
        pass



