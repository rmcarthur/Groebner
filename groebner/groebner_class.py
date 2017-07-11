from operator import itemgetter
import itertools
import numpy as np
from groebner import maxheap
import math
from groebner.multi_cheb import MultiCheb
from groebner.multi_power import MultiPower
from scipy.linalg import lu, qr, solve_triangular
from groebner.maxheap import Term
import matplotlib.pyplot as plt

#If clean is true then at a couple of places (end of rrqr_reduce and end of add r to matrix) things close to 0 will be made 0.
#Might make it more stable, might make it less stable. Not sure.

class Groebner(object):

    def __init__(self,polys):
        '''
        polys -- a list of polynomials that generate your ideal
        self.old_polys - The polynomials that have already gone through the solve loop once. Starts as none.
        self.new_polys - New polynomials that have never been through the solve loop. All of them at first.
        self.np_matrix - The full matrix of polynomials.
        self.term_set - The set of monomials in the matrix. Contains Terms.
        self.lead_term_set - The set of monomials that are lead terms of some polynomial in the matrix. Contains Terms.
        These next three are used to determine what polynomials to keep after reduction.
        self.original_lms - The leading Terms of the original polynomials (not phi's or r's). Keep these as old_polys.
        self.original_lm_dict - A dictionary of the leading terms to their polynomials
        self.not_needed_lms - The leading terms that have another leading term that divided them. We won't keep these.
        self.duplicate_lms - The leading terms that occur multiple times. Keep these as old_polys
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
        self.new_polys = polys
        self.np_matrix = np.array([])
        self.term_set = set()
        self.lead_term_set = set()
        self.original_lms = set()
        self.original_lm_dict = {}
        self.not_needed_lms = set()
        self.duplicate_lms = set()
        # this determines what is considered zero
        self.global_accuracy = 1.e-10

    def initialize_np_matrix(self):
        '''
        Initialzes self.np_matrix to having just old_polys and new_polys in it
        matrix_terms is the header of the matrix, it lines up each column with a monomial
        '''
        self.matrix_terms = []
        self.np_matrix = np.array([])
        self.term_set = set()
        self.lead_term_set = set()
        self.original_lm_dict = {}
        self.original_lms = set()
        self.not_needed_lms = set()
        self.duplicate_lms = set()

        for p in self.new_polys + self.old_polys:
            if p.lead_term != None:
                self.original_lms.add(Term(p.lead_term))
                self.original_lm_dict[Term(p.lead_term)] = p

        self._add_polys(self.new_polys + self.old_polys)

    def solve(self, qr_reduction = True):
        '''
        The main function. Initializes the matrix, adds the phi's and r's, and then reduces it. Repeats until the reduction
        no longer adds any more polynomials to the matrix. Print statements let us see the progress of the code.
        '''
        polys_were_added = True
        i=1 #Tracks what loop we are on.
        while polys_were_added:
            #print("Starting Loop #"+str(i))
            #print("Num Polys - ", len(self.new_polys + self.old_polys))
            #print("Initializing")
            self.initialize_np_matrix()
            #print(self.np_matrix.shape)
            #print("ADDING PHI's")
            self.add_phi_to_matrix()
            #print(self.np_matrix.shape)
            #print("ADDING r's")
            self.add_r_to_matrix()
            #print(self.np_matrix.shape)
            polys_were_added = self.reduce_matrix(qr_reduction = qr_reduction)
            i+=1

        #print("WE WIN")
        #print("Basis - ")
        return self.reduce_groebner_basis()

    def reduce_groebner_basis(self):
        '''
        Turns the groebner basis into a reduced groebner basis
        '''
        groebner_basis = list()
        #Checks if the polynomial 1 is in the basis. If so, this is the basis. This reduction won't happen earlier
        #becasue of the phi criterion check. Maybe.
        hasOne = False
        for poly in self.old_polys:
            if all([i==1 for i in poly.coeff.shape]):
                hasOne = True
            if np.sum(np.sum(abs(poly.coeff))) > self.global_accuracy:
                groebner_basis.append(poly)
        if hasOne:
            groebner_basis = list()
            for poly in self.old_polys:
                if all([i==1 for i in poly.coeff.shape]):
                    groebner_basis.append(poly)
                    break
        #groebner_basis = self.reduce_polys(groebner_basis)
        for p in groebner_basis:
            print(p.coeff)
        return groebner_basis

    def sort_matrix(self):
        '''
        Sorts the matrix into degrevlex order.
        '''
        argsort_list, self.matrix_terms = self.argsort(self.matrix_terms)
        self.np_matrix = self.np_matrix[:,argsort_list]

    def argsort(self, index_list):
        '''
        Returns an argsort list for the index, as well as sorts the list in place
        '''
        argsort_list = sorted(range(len(index_list)), key=index_list.__getitem__)[::-1]
        index_list.sort()
        return argsort_list, index_list[::-1]

    def clean_matrix(self):
        '''
        Gets rid of rows and columns in the np_matrix that are all zero.
        '''
        ##This would replace all small values in the matrix with 0.
        self.np_matrix[np.where(abs(self.np_matrix) < self.global_accuracy)]=0

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

    def sm_to_poly(self,rows,reduced_matrix):
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
        for i in rows:
            p = reduced_matrix[i]
            p[np.where(abs(p) < self.global_accuracy)] = 0
            coeff = np.zeros(shape)
            for j,term in enumerate(matrix_term_vals):
                coeff[term] = p[j]

            if self.power:
                poly = MultiPower(coeff)
                p_list.append(poly)
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
        self.lead_term_set.add(Term(p.lead_term))

        #Adds a new row of 0's if the matrix has any width
        if(self.np_matrix.shape[0] != 0):
            zero_poly = np.zeros((1,self.np_matrix.shape[1]))
            self.np_matrix = np.vstack((self.np_matrix,zero_poly))

        for idx in zip(*np.where(p.coeff != 0)):
            idx_term = Term(tuple(idx)) #Get a term object
            # Grab each non-zero element, put it into matrix.
            idx_term.val = tuple(map(lambda i: int(i), idx_term.val))
            coeff_val = p.coeff[idx_term.val]

            # If already in idx_list
            if idx_term in self.term_set:
                # get index of label and np matrix to put into
                idx_where = np.argmax([i.val == idx_term.val for i in self.matrix_terms])
                self.np_matrix[-1,idx_where] = coeff_val

            # If new column needed
            else:
                # Make new column
                self.term_set.add(idx_term)
                #If r's being added, adds new monomial to the heap
                if adding_r:
                    if(idx_term not in self.lead_term_set):
                        self.monheap.heappush(idx_term)
                        #print(idx_term.val)
                length_of_matrix = self.np_matrix.shape[0]
                if length_of_matrix == 0:
                    self.np_matrix = np.zeros((1,1))
                else:
                    zeros = np.zeros((length_of_matrix,1))
                    self.np_matrix = np.hstack((self.np_matrix, zeros))
                self.matrix_terms.append(idx_term)
                self.np_matrix[-1,-1] = coeff_val

    def _add_polys(self, p_list):
        '''
        p_list - a list of polynomials
        Adds the polynomials to self.np_matrix
        '''
        for p in p_list:
            self._add_poly_to_matrix(p)

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

        #Keeping track of lead_terms
        if sum(a_diff)==0 and sum(b_diff)==0:
            self.duplicate_lms.add(Term(a.lead_term))
        elif sum(a_diff)==0:
            self.not_needed_lms.add(Term(a.lead_term))
        elif sum(b_diff)==0:
            self.not_needed_lms.add(Term(b.lead_term))

        return a.mon_mult(a_diff), b.mon_mult(b_diff)

    def add_phi_to_matrix(self,phi = True):
        '''
        Takes all new possible combinations of phi polynomials and adds them to the Groebner Matrix
        Includes some checks to throw out unnecessary phi's
        '''

        # Find the set of all pairs of index the function will run through

        # Index_new iterate the tuple of every combination of the new_polys.
        index_new = itertools.combinations(range(len(self.new_polys)),2)
        # Index_oldnew iterates the tuple of every combination of new and old polynomials
        index_oldnew = itertools.product(range(len(self.new_polys)),range(len(self.new_polys),
                                               len(self.old_polys)+len(self.new_polys)))
        B = set(itertools.chain(index_new,index_oldnew))

        # Iterating through both possible combinations.
        while B:
            i,j = B.pop()
            if self.phi_criterion(i,j,B,phi)== True:
                #calculate the phi's.
                poly = self.new_polys + self.old_polys
                p_a , p_b = self.calc_phi(poly[i],poly[j])
                # add the phi's on to the Groebner Matrix.
                self._add_poly_to_matrix(p_a)
                self._add_poly_to_matrix(p_b)


    def phi_criterion(self,i,j,B,phi):
        # Need to run tests
        '''
        Parameters:
        i (int) : the index of the first polynomial
        j (int) : the index of the second polynomial
        B (set) : index of the set of polynomials to be considered.

        Returns:
           (bool) : returns False if
                1) The polynomials at index i and j are relative primes or
                2) there exists an l such that (i,l) or (j,l) will not be considered in
                the add_phi_to_matrix() method and LT(l) divides lcm(LT(i),LT(j)),
                otherwise, returns True.
           * See proposition 8 in "Section 10: Improvements on Buchburger's algorithm."
       '''
        if phi == False:
            return True
        # List of new and old polynomials.
        polys = self.new_polys+self.old_polys
        #Always add these?, they are helping to reduce our basis.< I just ran some tests, the timing is about the same.
        #if all(polys[j].lead_term == self._lcm(polys[i],polys[j])) or all(polys[i].lead_term == self._lcm(polys[i],polys[j])) :
        #    return True



        # Relative Prime check: If the lead terms of i and j are relative primes, phi is not needed
        if all([a*b == 0 for a,b in zip(polys[i].lead_term,polys[j].lead_term)]):
            return False

        # Another criterion
        else:
            for l in range(len(polys)):
                #print ("For l = {}:".format(l))

                # Checks that l is not j or i.
                if l == j or l == i:
                    #print("\t{} is i or j".format(l))
                    continue

                # Sorts the tuple (i,l) or (l,i) in order of smaller to bigger.
                i_tuple = tuple(sorted((i,l)))
                j_tuple = tuple(sorted((j,l)))

                # i_tuple and j_tuple needs to not be in B.
                if j_tuple in B or i_tuple in B:
                    #print('\t{} or {} is in B'.format(j_tuple,i_tuple))
                    continue

                lcm = self._lcm(polys[i],polys[j])
                lead_l = polys[l].lead_term

                # See if LT(poly[l]) divides lcm(LT(i),LT(j))
                if all([i-j>=0 for i,j in zip(lcm,lead_l)]) :
                    #print("\tLT of poly[l] divides lcm(LT(i),LT(j)")
                    return False

        # Function will return True and calculate phi if none of the checks passed for all l's.

            return True


    def _build_maxheap(self):
        '''
        Builds a maxheap for use in r polynomial calculation
        '''
        self.monheap = maxheap.MaxHeap()
        for mon in self.term_set:
            if(mon not in self.lead_term_set): #Adds every monomial that isn't a lead term to the heap
                self.monheap.heappush(mon)

    def calc_r(self, m):
        '''
        Finds the r polynomial that has a leading monomial m
        Returns the polynomial.
        '''
        for p in self.new_polys + self.old_polys:
                l = list(p.lead_term)
                if all([i<=j for i,j in zip(l,m)]) and len(l) == len(m): #Checks to see if l divides m
                    c = [j-i for i,j in zip(l,m)]
                    if not l == m: #Make sure c isn't all 0
                        return p.mon_mult(c)
        if self.power:
            return MultiPower(np.array([0]))
        else:
            return MultiCheb(np.array([0]))

    def add_r_to_matrix(self,clean=False):
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
        if clean:
            self.clean_matrix()

    def reduce_matrix(self, qr_reduction=True):
        '''
        Reduces the matrix fully using either QR or LU decomposition. Adds the new_poly's to old_poly's, and adds to
        new_poly's any polynomials created by the reduction that have new leading monomials.
        Returns-True if new polynomials were found, False otherwise.
        '''
        if qr_reduction:
            #Get a full rank submatrix.
            Q,R,P = qr(self.np_matrix, pivoting = True) #rrqr reduce it
            PT = self.inverse_P(P)
            diagonals = np.diagonal(R) #Go along the diagonals to find the rank
            rank = np.sum(np.abs(diagonals)>self.global_accuracy)
            reorder = R[:,PT]
            full_rank = reorder[:rank,]

            ##I think we could ignore the getting full rank and just use this line, as the full length would be found in
            ##the recursion. We would just have to remove the bottom zeros before passing to triangular solve.
            reduced_matrix = self.rrqr_reduce(full_rank)
            reduced_matrix = self.triangular_solve(reduced_matrix)
            print(reduced_matrix)
            #plt.matshow([i==0 for i in reduced_matrix])
            #plt.matshow([abs(i)<self.global_accuracy for i in reduced_matrix])
        else:
            P,L,U = lu(self.np_matrix)
            reduced_matrix = U
            reduced_matrix = self.fully_reduce(reduced_matrix, qr_reduction = False)

        #Get the new polynomials
        new_poly_spots = list()
        old_poly_spots = list()

        already_looked_at = set() #rows whose leading monomial we've already checked
        for i, j in zip(*np.where(reduced_matrix==1)):
            if i in already_looked_at: #We've already looked at this row
                continue
            elif self.matrix_terms[j] in self.lead_term_set: #The leading monomial is not new.
                if self.matrix_terms[j] in self.original_lms - (self.not_needed_lms - self.duplicate_lms): #Reduced old poly
                    old_poly_spots.append(i)
                    self.original_lm_dict.pop(self.matrix_terms[j], None)
                already_looked_at.add(i)
                continue
            else:
                already_looked_at.add(i)
                new_poly_spots.append(i) #This row gives a new leading monomial

        self.old_polys = self.sm_to_poly(old_poly_spots, reduced_matrix)
        self.new_polys = self.sm_to_poly(new_poly_spots, reduced_matrix)

        #If any of the ones we need reduced out fully, put them back in
        for i in self.original_lms - (self.not_needed_lms - self.duplicate_lms):
            if i in self.original_lm_dict:
                self.old_polys.append(self.original_lm_dict[i])

        return len(self.new_polys) > 0

    def clean_zeros_from_matrix(self,matrix):
        '''
        Gets rid of rows and columns in the np_matrix that are all zero.
        '''
        ##This would replace all small values in the matrix with 0.
        matrix[np.where(abs(matrix) < self.global_accuracy)]=0
        return matrix

    def rrqr_reduce(self, matrix, clean=False):
        if matrix.shape[0]==0 or matrix.shape[1]==0:
            return matrix
        height = matrix.shape[0]
        A = matrix[:height,:height] #Get the square submatrix
        B = matrix[:,height:] #The rest of the matrix to the right
        Q,R,P = qr(A, pivoting = True) #rrqr reduce it
        PT = self.inverse_P(P)
        diagonals = np.diagonal(R) #Go along the diagonals to find the rank
        rank = np.sum(np.abs(diagonals)>self.global_accuracy)
        if rank == height: #full rank, do qr on it
            Q,R = qr(A)
            A = R #qr reduce A
            B = Q.T.dot(B) #Transform B the same way
        else: #not full rank
            A = R[:,PT] #Switch the columns back
            B = Q.T.dot(B) #Multiply B by Q transpose
            #sub1 is the top part of the matrix, we will recursively reduce this
            #sub2 is the bottom part of A, we will set this all to 0
            #sub3 is the bottom part of B, we will recursively reduce this.
            #All submatrices are then put back in the matrix and it is returned.
            sub1 = np.hstack((A[:rank,],B[:rank,])) #Takes the top parts of A and B
            result = self.rrqr_reduce(sub1) #Reduces it
            A[:rank,] = result[:,:height] #Puts the A part back in A
            B[:rank,] = result[:,height:] #And the B part back in B

            sub2 = A[rank:,]
            zeros = np.zeros_like(sub2)
            A[rank:,] = np.zeros_like(sub2)

            sub3 = B[rank:,]
            B[rank:,] = self.rrqr_reduce(sub3)

        reduced_matrix = np.hstack((A,B))

        if not clean:
            return reduced_matrix
        else:
            return self.clean_zeros_from_matrix(reduced_matrix)

    def inverse_P(self,p):
        '''
        Takes in the one dimentional array of column switching.
        Returns the one dimentional array of switching it back.
        '''
        # The elementry matrix that flips the columns of given matrix.
        P = np.eye(len(p))[:,p]
        # This finds the index that equals 1 of each row of P.
        #(This is what we want since we want the index of 1 at each column of P.T)
        return np.where(P==1)[1]

    def triangular_solve(self,matrix):
        " Reduces the upper block triangular matrix. "
        m,n = matrix.shape
        j = 0  # The row index.
        k = 0  # The column index.
        c = [] # It will contain the columns that make an upper triangular matrix.
        d = [] # It will contain the rest of the columns.
        order_c = [] # List to keep track of original index of the columns in c.
        order_d = [] # List to keep track of the original index of the columns in d.

        # Checks if the given matrix is not a square matrix.
        if m != n:
            # Makes sure the indicies are within the matrix.
            while j < m and k < n:
                if matrix[j,k]!= 0:
                    c.append(matrix[:,k])
                    order_c.append(k)
                    # Move to the diagonal if the index is non-zero.
                    j+=1
                    k+=1
                else:
                    d.append(matrix[:,k])
                    order_d.append(k)
                    # Check the next column in the same row if index is zero.
                    k+=1
            # C will be the square matrix that is upper triangular with no zeros on the diagonals.
            C = np.vstack(c).T
            # If d is not empty, add the rest of the columns not checked into the matrix.
            if d:
                D = np.vstack(d).T
                D = np.hstack((D,matrix[:,k:]))
            else:
                D = matrix[:,k:]
            # Append the index of the rest of the columns to the order_d list.
            for i in range(n-k):
                order_d.append(k)
                k+=1

            # Solve for the CX = D
            X = solve_triangular(C,D)

            # Add I to X. [I|X]
            solver = np.hstack((np.eye(X.shape[0]),X))

            # Find the order to reverse the columns back.
            order = self.inverse_P(order_c+order_d)

            # Reverse the columns back.
            solver = solver[:,order]
            # Temporary checker. Plots the non-zero part of the matrix.
            #plt.matshow(~np.isclose(solver,0))

            return solver

        else:
        # The case where the matrix passed in is a square matrix
            return np.eye(m)


#    """
#    Functions we once used but don't anymore, not sure if we want to delete them yet.
#
#    def fully_reduce(self, matrix, qr_reduction = True):
#        '''
#        WE NO LONGER USE THIS FUNCTION
#        Fully reduces the matrix by making sure all submatrices formed by taking out columns of zeros are
#        also in upper triangular form. Does this recursively. Returns the reduced matrix.
#        '''
#        matrix = self.clean_zeros_from_matrix(matrix)
#        diagonals = np.diagonal(matrix).copy()
#        zero_diagonals = np.where(abs(diagonals)==0)[0]
#        if(len(zero_diagonals != 0)):
#            first_zero = zero_diagonals[0]
#            i = first_zero
#            #Checks how many rows we can go down that are all 0.
#            while all([k==0 for k in matrix[first_zero:,i:i+1]]):
#                i+=1
#                if(i == matrix.shape[1]):
#                    i = -1
#                    break
#                pass
#
#            if(i != -1):
#                sub_matrix = matrix[first_zero: , i:]
#                if qr_reduction:
#                    Q,R = qr(sub_matrix)
#                    sub_matrix = self.fully_reduce(R)
#                else:
#                    P,L,U = lu(sub_matrix)
#                    #ERROR HERE BECAUSE OF THE PERMUATION MATRIX, I'M NOT SURE HOW TO FIX IT
#                    sub_matrix = self.fully_reduce(U, qr_reduction = False)
#
#                matrix[first_zero: , i:] = sub_matrix
#        return self.clean_zeros_from_matrix(matrix)
#
#    def pad_back(self,mon,poly):
#        tuple1 = []
#        for i in mon:
#            list1 = (0,i)
#            tuple1.append(list1)
#        if self.power:
#            return MultiPower(np.pad(poly.coeff, tuple1, 'constant', constant_values = 0), clean_zeros = False)
#        else:
#            return MultiCheb(np.pad(poly.coeff, tuple1, 'constant', constant_values = 0), clean_zeros = False)
#
#    def reduce_polys(self, polys):
#        '''
#        WE NO LONGER USE THIS FUNCTION AS FAR AS I KNOW
#        reduces the given list of polynomials and returns the non-zero ones
#        '''
#        change = True
#        while change:
#            change = False
#            for poly, other in itertools.permutations(polys,2):
#                if poly.lead_term == None or other.lead_term == None:
#                    continue #one of them is empty
#                if other != poly and all([i-j >= 0 for i,j in zip(poly.lead_term,other.lead_term)]):
#                    monomial = tuple(np.subtract(poly.lead_term,other.lead_term))
#                    new = other.mon_mult(monomial) #New polynomial with same lead_term as poly
#
#                    #Pad the polynomials so they have the same shape and can be subtracted
#                    lcm = np.maximum(poly.coeff.shape, new.coeff.shape)
#
#                    poly_pad = np.subtract(lcm, poly.coeff.shape)
#                    poly_pad[np.where(poly_pad<0)]=0
#                    pad_poly = self.pad_back(poly_pad, poly)
#
#                    new_pad = np.subtract(lcm, new.coeff.shape)
#                    new_pad[np.where(new_pad<0)]=0
#                    pad_new = self.pad_back(new_pad,new)
#
#                    new_coeff = pad_poly.coeff-pad_new.coeff
#                    new_coeff[np.where(abs(new_coeff) < self.global_accuracy)]=0 #Get rid of floating point errors to make more stable
#                    poly.__init__(new_coeff)
#                    #print(poly.coeff)
#                    change = True
#        non_zeros = list()
#        for p in polys:
#            p.coeff[np.where(abs(p.coeff) < self.global_accuracy)]=0
#            if p.lead_term==None or p in non_zeros:
#                continue
#            non_zeros.append(p)
#        return non_zeros
#
#    def reduce_poly(self, poly):
#        '''
#        WE NO LONGER USE THIS EITHER
#        Divides a polynomial by the polynomials we already have to see if it contains any new info
#        '''
#        change = True
#        while change:
#            change = False
#            for other in self.old_polys:
#                if poly.lead_term == None or other.lead_term == None:
#                    continue #one of them is empty
#                if other != poly and all([i-j >= 0 for i,j in zip(poly.lead_term,other.lead_term)]):
#                    #print(poly.coeff)
#                    #print(other.coeff)
#                    monomial = tuple(np.subtract(poly.lead_term,other.lead_term))
#                    new = other.mon_mult(monomial)
#
#                    lcm = np.maximum(poly.coeff.shape, new.coeff.shape)
#
#                    poly_pad = np.subtract(lcm, poly.coeff.shape)
#                    poly_pad[np.where(poly_pad<0)]=0
#                    pad_poly = self.pad_back(poly_pad, poly)
#
#                    new_pad = np.subtract(lcm, new.coeff.shape)
#                    new_pad[np.where(new_pad<0)]=0
#                    pad_new = self.pad_back(new_pad,new)
#
#                    new_coeff = pad_poly.coeff-(poly.lead_coeff/other.lead_coeff)*pad_new.coeff
#                    new_coeff[np.where(abs(new_coeff) < self.global_accuracy)]=0 #Get rid of floating point errors to make more stable
#                    poly.__init__(new_coeff)
#                    change = True
#        return poly
#
#
#    """
