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

    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        super(MultiCheb, self).__init__(coeff, order, lead_term, clean_zeros)


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

    def _reverse_axes(self):
        """
        Reverse the axes of the coeff tensor.
        """
        return self.coeff.flatten()[::-1].reshape(self.coeff.shape)


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
        a = MultiCheb(np.pad(a.coeff,add_a_list.astype(int),'constant'), clean_zeros = False)
        b = MultiCheb(np.pad(b.coeff,add_b_list.astype(int),'constant'), clean_zeros = False)
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
        c = new_other._reverse_axes()
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

    def fold_in_i_dir(solution_matrix, dim, i, x, fold_idx):
        sol = np.zeros_like(solution_matrix)
        slice_0 = slice(None, 1, None)
        slice_1 = slice(fold_idx, fold_idx+1, None)

        indexer1 = [slice(None)]*dim
        indexer2 = [slice(None)]*dim
        indexer3 = [slice(None)]*dim

        indexer1[i] = slice_0
        indexer2[i] = slice_1

        sol[indexer1] = solution_matrix[indexer2]
    
        for n in range(x):

            slice_2 = slice(n+1, n+2, None)
            slice_3 = slice(fold_idx+n+1, fold_idx+n+2, None)
            slice_4 = slice(fold_idx-n-1, fold_idx-n, None)

            indexer1[i] = slice_2
            indexer2[i] = slice_3
            indexer3[i] = slice_4

            if fold_idx-n-1 < 0:
                if fold_idx+n+2 > x:
                    break
                else:
                    sol[indexer1] = solution_matrix[indexer2]
            else:
                if fold_idx+n+2 > x:
                    sol[indexer1] = solution_matrix[indexer3]
                else:
                    sol[indexer1] = solution_matrix[indexer3] + solution_matrix[indexer2]

        return sol


    def mon_mult(self,idx):
    
        pad_values = list()
        for i in idx: #iterates through monomial and creates a tuple of pad values for each dimension
            pad_dim_i = (i,0)
            #In np.pad each dimension is a tuple of (i,j) where i is how many to pad in front and j is how many to pad after.
            pad_values.append(pad_dim_i)
        p1 = np.pad(self, (pad_values), 'constant', constant_values = 0)

        number_of_dim = self.ndim
        shape_of_self = self.shape
        solution_matrix = self

        for i in range(number_of_dim):
            solution_matrix = MultiCheb.fold_in_i_dir(solution_matrix, number_of_dim, i, shape_of_self[i], idx[i])

        big = p1.shape
        little = solution_matrix.shape
    
        pad_list = list()
        for i,j in zip(big,little):
            z = i-j
            pad_list.append((0,z))

        p2 = np.pad(solution_matrix, (pad_list), 'constant', constant_values = 0)
        Pf = (p1+p2)
        Pf = .5*Pf
        return MultiCheb(Pf)

