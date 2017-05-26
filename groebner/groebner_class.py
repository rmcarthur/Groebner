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
        self.old_polys - The polynomials that have already gone through the solve loop once. Starts as none.
        self.new_polys - New polynomials that have never been throught the solve loop. All of them at first.
        self.np_matrix - The full matrix of polynomials.
        self.term_set - The set of monomials in the matrix.
        self.lead_term_set - The set of monomials that are lead terms of some polynomial in the matrix.
        '''
        # Check polynomial types
        if all([type(p) == MultiPower for p in polys]):
            self.power = True
        elif all([type(p) == MultiCheb for p in polys]):
            self.power = False
        else:
            print([type(p) == MultiPower for p in polys])
            raise ValueError('Bad polynomials in list')

        self.old_polys = list()
        self.new_polys = self.reduce_polys(polys)
        self.np_matrix = np.array([])
        self.term_set = set()
        self.lead_term_set = set()
        #for p in self.new_polys:
        #    print(p.coeff)

    def initialize_np_matrix(self):
        '''
        Initialzes self.np_matrix to having just old_polys and new_polys in it
        matrix_terms is the header of the matrix, it lines up each column with a monomial
        '''
        self.matrix_terms = []
        self.np_matrix = np.array([])
        self.term_set = set()
        self.lead_term_set = set()

        self.new_polys = self.reduce_polys(self.new_polys+self.old_polys)
        self.old_polys = list()

        self._add_polys(self.new_polys)
        self._add_polys(self.old_polys)
        self.clean_matrix()
        pass

    def solve(self, qr_reduction = True):
        '''
        The main function. Initializes the matrix, adds the phi's and r's, and then reduces it. Repeats until the reduction
        no longer adds any more polynomials to the matrix. Print statements let us see the progress of the code.
        '''
        polys_were_added = True
        i=1 #Tracks what loop we are on.
        while polys_were_added:
            print("Starting Loop #"+str(i))
            print("Initializing")
            self.initialize_np_matrix()
            print(self.np_matrix.shape)
            print("ADDING PHI's")
            self.add_phi_to_matrix()
            print(self.np_matrix.shape)
            print("ADDING r's")
            self.add_r_to_matrix()
            print(self.np_matrix.shape)
            polys_were_added = self.reduce_matrix(qr_reduction = qr_reduction)
            i+=1
            #if i > 1:
            #    polys_were_added = False
            #    self.old_polys += self.new_polys
        print("WE WIN")
        return self.reduce_groebner_basis()
        pass

    def pad_back(self,mon,poly):
        tuple1 = []
        for i in mon:
            list1 = (0,i)
            tuple1.append(list1)
        if self.power:
            return MultiPower(np.pad(poly.coeff, tuple1, 'constant', constant_values = 0), clean_zeros = False)
        else:
            return MultiCheb(np.pad(poly.coeff, tuple1, 'constant', constant_values = 0), clean_zeros = False)

    def reduce_polys(self, polys):
        """
        reduces the given list of polynomials and returns the non-zero ones
        """
        change = True
        while change:
            change = False
            for poly in polys:
                for other in polys:
                    if poly.lead_term == None or other.lead_term == None:
                        continue #one of them is empty
                    if other != poly and all([i-j >= 0 for i,j in zip(poly.lead_term,other.lead_term)]):
                        monomial = tuple(np.subtract(poly.lead_term,other.lead_term))
                        new = other.mon_mult(monomial)

                        lcm = np.maximum(poly.coeff.shape, new.coeff.shape)

                        poly_pad = np.subtract(lcm, poly.coeff.shape)
                        poly_pad[np.where(poly_pad<0)]=0
                        pad_poly = self.pad_back(poly_pad, poly)

                        new_pad = np.subtract(lcm, new.coeff.shape)
                        new_pad[np.where(new_pad<0)]=0
                        pad_new = self.pad_back(new_pad,new)

                        new_coeff = pad_poly.coeff-(poly.lead_coeff/other.lead_coeff)*pad_new.coeff
                        new_coeff[np.where(abs(new_coeff) < 1.e-10)]=0 #Get rid of floating point errors to make more stable
                        poly.__init__(new_coeff)
                        #print(poly.coeff)
                        change = True
                        pass
                    pass
                pass
            pass
        non_zeros = list()
        for p in polys:
            p.coeff[np.where(abs(p.coeff) < 1.e-15)]=0
            if p.lead_term==None or p in non_zeros:
                continue
            non_zeros.append(p)
            pass
        return non_zeros

    def reduce_poly(self, poly, divisors=[]):
        """
        Divides a polynomial by the polynomials we already have to see if it contains any new info
        """
        if not divisors:
            divisors = self.old_polys
        change = True
        while change:
            change = False
            for other in divisors:
                if poly.lead_term == None or other.lead_term == None:
                    continue #one of them is empty
                if other != poly and all([i-j >= 0 for i,j in zip(poly.lead_term,other.lead_term)]):
                    #print(poly.coeff)
                    #print(other.coeff)
                    monomial = tuple(np.subtract(poly.lead_term,other.lead_term))
                    new = other.mon_mult(monomial)

                    lcm = np.maximum(poly.coeff.shape, new.coeff.shape)

                    poly_pad = np.subtract(lcm, poly.coeff.shape)
                    poly_pad[np.where(poly_pad<0)]=0
                    pad_poly = self.pad_back(poly_pad, poly)

                    new_pad = np.subtract(lcm, new.coeff.shape)
                    new_pad[np.where(new_pad<0)]=0
                    pad_new = self.pad_back(new_pad,new)

                    new_coeff = pad_poly.coeff-(poly.lead_coeff/other.lead_coeff)*pad_new.coeff
                    new_coeff[np.where(abs(new_coeff) < 1.e-10)]=0 #Get rid of floating point errors to make more stable
                    poly.__init__(new_coeff)
                    print("Poly coeff: ", poly.coeff)
                    #print(poly.coeff)
                    change = True
                    print("after true: ", change)
                    pass
                pass
            pass
        return poly

    def reduce_groebner_basis(self):
        '''
        Turns the groebner basis into a reduced groebner basis
        '''
        groebner_basis = list()
        for poly in self.old_polys:
            if np.sum(np.sum(abs(poly.coeff))) > 1.e-10:
                groebner_basis.append(poly)
                pass
            pass
        groebner_basis = self.reduce_polys(groebner_basis)
        for p in groebner_basis:
            print(p.coeff)
            pass
        return groebner_basis

    def sort_matrix(self):
        '''
        Sorts the matrix into degrevlex order.
        '''
        argsort_list, self.matrix_terms = self.argsort(self.matrix_terms)
        self.np_matrix = self.np_matrix[:,argsort_list]
        pass

    def clean_matrix(self):
        '''
        Gets rid of rows and columns in the np_matrix that are all zero.
        '''
        ##This would replace all small values in the matrix with 0.
        self.np_matrix[np.where(abs(self.np_matrix) < 1.e-10)]=0

        #Removes all 0 monomials
        non_zero_monomial = np.sum(abs(self.np_matrix), axis=0)>0 ##Increasing this will get rid of small things.
        self.np_matrix = self.np_matrix[:,non_zero_monomial] #only keeps the non_zero_monomials
        #If a monomial was removed, removes it from self.matrix_terms  as well, and the term set
        to_remove = set()
        for i,j in zip(self.matrix_terms, non_zero_monomial):
            if not j:
                to_remove.add(i)
        for i in to_remove:
            self.matrix_terms.remove(i)
            self.term_set.remove(i)
        #Removes all 0 polynomials
        non_zero_polynomial = np.sum(abs(self.np_matrix),axis=1)>0 ##Increasing this will get rid of small things.
        self.np_matrix = self.np_matrix[non_zero_polynomial,:] #Only keeps the non_zero_polymonials
        pass

    def sm_to_poly(self,idxs,reduced_matrix):
        '''
        Takes a list of indicies corresponding to the rows of the reduced matrix and
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

            if self.power:
                poly = MultiPower(coeff)
            else:
                poly = MultiCheb(coeff)

            p_list.append(poly)
        return p_list

    def _add_poly_to_matrix(self, p, adding_r = False):
        '''
        Takes in a single polynomial and adds it to the state matrix
        First adds a row of zeros, then goes through each monomial in the polynomial and puts it's coefficient in
        adding new columns as needed when new monomials are added.

        adding_r is only true when the r's are being added, this way it knows to keep adding new monomials to the heap
        for further r calculation
        '''
        self.lead_term_set.add(p.lead_term)

        #Adds a new row of 0's if the matrix has any width
        if(self.np_matrix.shape[0] != 0):
            zero_poly = np.zeros((1,self.np_matrix.shape[1]))
            self.np_matrix = np.vstack((zero_poly,self.np_matrix))

        for idx in p.degrevlex_gen():
            idx_term = maxheap.Term(tuple(idx)) #Get a term object
            # Grab each non-zero element, put it into matrix.
            idx_term.val = tuple(map(lambda i: int(i), idx_term.val))
            coeff_val = p.coeff[idx_term.val]

            if(coeff_val == 0):
                continue
            # If already in idx_list
            if idx_term in self.term_set:
                # get index of label and np matrix to put into
                idx_where = np.argmax([i == idx_term for i in self.matrix_terms])
                self.np_matrix[0,idx_where] = coeff_val

            # If new column needed
            else:
                # Make new column
                self.term_set.add(idx_term)
                #If r's being added, adds new monomial to the heap
                if adding_r:
                    if(idx_term.val not in self.lead_term_set):
                        self.monheap.heappush(idx_term)
                length_of_matrix = self.np_matrix.shape[0]
                if length_of_matrix == 0:
                    self.np_matrix = np.zeros((1,1))
                else:
                    zeros = np.zeros((length_of_matrix,1))
                    self.np_matrix = np.hstack((self.np_matrix, zeros))
                self.matrix_terms.append(idx_term)
                self.np_matrix[0,-1] = coeff_val
        pass

    def _add_polys(self, p_list):
        '''
        p_list - a list of polynomials
        Adds the polynomials to self.np_matrix
        '''
        for p in p_list:
            self._add_poly_to_matrix(p)
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
        Includes some checks to throw out unnecessary phi's
        '''
        for i,j in itertools.combinations(self.new_polys+self.old_polys,2):
            # This prevents calculation of phi with combinations of old_f exclusively.
            if i not in self.old_polys:
                # Relative prime check: If the elementwise multiplication of list i and j are all zeros, calculation of phi is not needed.
                #(I separated the if statements for better visibility reasons, if it's better to combine, please fix!)
                if not all([ a == 0 or b ==0 for a,b in zip(i.lead_term, j.lead_term)]):
                # Calculate the phi's.
                    p_a , p_b = self.calc_phi(i,j)
                    # Add the phi's on to the Groebner Matrix.
                    self._add_poly_to_matrix(p_a)
                    self._add_poly_to_matrix(p_b)
        self.clean_matrix()
        pass

    def _build_maxheap(self):
        '''
        Builds a maxheap for use in r polynomial calculation
        '''
        self.monheap = maxheap.MaxHeap()
        for mon in self.term_set:
            if(mon.val not in self.lead_term_set): #Adds every monomial that isn't a lead term to the heap
                self.monheap.heappush(mon)
        pass

    def calc_r(self, m):
        '''
        Finds the r polynomial that has a leading monomial m
        Returns the polynomial.
        '''
        for p in self.new_polys + self.old_polys:
                l = list(p.lead_term)
                if all([i<=j for i,j in zip(l,m)]) and len(l) == len(m): #Checks to see if l divides m
                    #New mon_mult method. Will break with MultiCheb unitl that is added.
                    c = [j-i for i,j in zip(l,m)]
                    if not l == m: #Make sure c isn't all 0
                        return p.mon_mult(c)
        return MultiPower(np.array([0]))

    def add_r_to_matrix(self):
        '''
        Finds the r polynomials and adds them to the matrix.
        First makes Heap out of all potential monomials, then finds polynomials with leading terms that divide it and
        add them to the matrix.
        '''
        self._build_maxheap()
        while len(self.monheap) > 0:
            m = list(self.monheap.heappop().val)
            r = self.calc_r(m)
            if not r.lead_term==None:
                self._add_poly_to_matrix(r, adding_r = True)
        self.sort_matrix()
        self.clean_matrix()
        pass

    def reduce_matrix(self, qr_reduction=True):
        '''
        Reduces the matrix fully using either QR or LU decomposition. Adds the new_poly's to old_poly's, and adds to
        new_poly's any polynomials created by the reduction that have new leading monomials.
        Returns-True if new polynomials were found, False otherwise.
        '''
        di={}
        for i, j in zip(*np.where(self.np_matrix!=0)):
            if i in di:
                continue
            else:
                di[i]=j
        old_lms = set(di.values())

        #print(self.np_matrix)

        if qr_reduction:
            Q,R = qr(self.np_matrix)
            reduced_matrix = R
            reduced_matrix = self.fully_reduce(reduced_matrix)
        else:
            P,L,U = lu(self.np_matrix)
            reduced_matrix = U
            reduced_matrix = self.fully_reduce(reduced_matrix, qr_reduction = False)

        #Checks that it's fully reduced
        #print(reduced_matrix)
        reduced_matrix = self.fully_reduce(reduced_matrix)
        #print(reduced_matrix)

        #Get the new polynomials
        good_poly_spots = list()
        already_looked_at = set() #rows whose leading monomial we've already checked
        for i, j in zip(*np.where(reduced_matrix!=0)):
            if i in already_looked_at:
                continue
            elif j in old_lms:
                already_looked_at.add(i)
                continue
            else:
                already_looked_at.add(i)
                good_poly_spots.append(i) #This row gives a new leading monomial
        self.old_polys = self.new_polys + self.old_polys
        self.new_polys = list()
        new_polys = self.sm_to_poly(good_poly_spots, reduced_matrix)

        for p in new_polys:
            reduced_p = self.reduce_poly(p)
            if p.lead_term != None:
                self.new_polys.append(p)

        return len(self.new_polys) > 0


    def fully_reduce(self, matrix, qr_reduction = True):
        '''
        Fully reduces the matrix by making sure all submatrices formed by taking out columns of zeros are
        also in upper triangular form. Does this recursively. Returns the reduced matrix.
        '''
        matrix = self.clean_zeros_from_matrix(matrix)
        diagonals = np.diagonal(matrix).copy()
        zero_diagonals = np.where(abs(diagonals)==0)[0]
        if(len(zero_diagonals != 0)):
            first_zero = zero_diagonals[0]
            i = first_zero
            #Checks how many rows we can go down that are all 0.
            while all([k==0 for k in matrix[first_zero:,i:i+1]]):
                i+=1
                if(i == matrix.shape[1]):
                    i = -1
                    break
                pass

            if(i != -1):
                sub_matrix = matrix[first_zero: , i:]
                if qr_reduction:
                    Q,R = qr(sub_matrix)
                    sub_matrix = self.fully_reduce(R)
                else:
                    P,L,U = lu(sub_matrix)
                    #ERROR HERE BECAUSE OF THE PERMUATION MATRIX, I'M NOT SURE HOW TO FIX IT
                    sub_matrix = self.fully_reduce(U, qr_reduction = False)

                matrix[first_zero: , i:] = sub_matrix
        return self.clean_zeros_from_matrix(matrix)
        #return matrix

    def clean_zeros_from_matrix(self,matrix):
        '''
        Gets rid of rows and columns in the np_matrix that are all zero.
        '''
        ##This would replace all small values in the matrix with 0.
        matrix[np.where(abs(matrix) < 1.e-10)]=0
        return matrix
        pass
