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

    def mon_mult(self, idx):
        '''Function creates a matrix of zeroes of the correct dimmension
        and then replaces the values in the matrix with the values given
        after adding corresponding terms

        To use this function pass the monomial in as a tuple corresponding to
        the values of the Chebychev polynomial. For example T_2(x)T_3(y) would
        be (2,3). This is idx and P is the polynomial in matrix form.
        '''
        original = self
        if len(idx) == 2:
            pad_values = list()
            for i in idx: #iterates through monomial and creates a tuple of pad values for each dimension
                pad_dim_i = (i,0)
                #In np.pad each dimension is a tuple of (i,j) where i is how many to pad in front and j is how many to pad after.
                pad_values.append(pad_dim_i)
            p1 = np.pad(original, (pad_values), 'constant', constant_values = 0)

            row,col  = idx
            num_rows,num_col = original.shape
            sol = np.zeros_like(original)
            sol[:1:,:] = original[row:row+1:,:]
            for n in range(num_rows):
                if row-n-1 < 0:
                    if row+n+2 > num_rows:
                        break
                    else:
                        sol[n+1:n+2:,::] = original[row+n+1:row+n+2:,::]
                else:
                    if row+n+2 > num_rows:
                        sol[n+1:n+2:,::] = original[row-n-1:row-n:,::]
                    else:
                        sol[n+1:n+2:,::] = original[row-n-1:row-n:,::] + original[row+n+1:row+n+2:,::]
            sol2 = np.zeros_like(original)
            sol2[::,:1:] = sol[::,col:col+1:]
            for n in range(num_col):
                if col-n-1 < 0:
                    if col+n+2 > num_col:
                        break
                    else:
                        sol2[::,n+1:n+2:] = sol[::,col+n+1:col+n+2:]
                else:
                    if col+n+2 > num_col:
                        sol2[::,n+1:n+2:] = sol[::,col-n-1:col-n:]
                    else:
                        sol2[::,n+1:n+2:] = sol[::,col-n-1:col-n:] + sol[::,col+n+1:col+n+2:]
            fsol_length, fsol_width = p1.shape
            sol2_length, sol2_width = sol2.shape
            add_length = fsol_length - sol2_length
            add_width = fsol_width - sol2_width
            p2 = np.pad(sol2, ((0,add_length),(0,add_width)), 'constant', constant_values = 0)
            Pf = (p1+p2)
            Pf = Pf.astype(float)
            Pf = .5*Pf
            return MultiCheb(Pf)


        elif len(idx) == 3:
            pad_values = list()
            for i in idx: #iterates through monomial and creates a tuple of pad values for each dimension
                pad_dim_i = (i,0)
                #In np.pad each dimension is a tuple of (i,j) where i is how many to pad in front and j is how many to pad after.
                pad_values.append(pad_dim_i)
            p1 = np.pad(original, (pad_values), 'constant', constant_values = 0)

            row,col,depth  = idx #row, col, depth correspond to the row, column, and depth of where the matrix will be folded.
            num_rows,num_col,num_z = original.shape #sets variables for the shape of thee polynomial input
            sol = np.zeros_like(original)
            sol[:1:,:,:] = original[row:row+1:,:,:]

            for n in range(num_rows):
                if row-n-1 < 0:
                    if row+n+2 > num_rows:
                        break
                    else:
                        sol[n+1:n+2:,:,:] = original[row+n+1:row+n+2:,:,:]
                else:
                    if row+n+2 > num_rows:
                        sol[n+1:n+2:,:,:] = original[row-n-1:row-n:,:,:]
                    else:
                        sol[n+1:n+2:,:,:] = original[row-n-1:row-n:,:,:] + original[row+n+1:row+n+2:,:,:]

            sol2 = np.zeros_like(original)
            sol2[::,:1:,:] = sol[::,col:col+1:,:]
            for n in range(num_col):
                if col-n-1 < 0:
                    if col+n+2 > num_col:
                        break
                    else:
                        sol2[::,n+1:n+2:,:] = sol[::,col+n+1:col+n+2:,:]
                else:
                    if col+n+2 > num_col:
                        sol2[::,n+1:n+2:,:] = sol[::,col-n-1:col-n:,:]
                    else:
                        sol2[::,n+1:n+2:,:] = sol[::,col-n-1:col-n:,:] + sol[::,col+n+1:col+n+2:,:]
            sol3 = np.zeros_like(original)
            sol3[:,:,:1:] = sol2[:,:,depth:depth+1:]
            for n in range(num_z):
                if depth-n-1 < 0:
                    if depth+n+2 > num_z:
                        break
                    else:
                        sol3[:,:,n+1:n+2:] = sol2[:,:,depth+n+1:depth+n+2:]
                else:
                    if depth+n+2 > num_z:
                        sol3[:,:,n+1:n+2:] = sol2[:,:,depth-n-1:depth-n:]
                    else:
                        sol3[:,:,n+1:n+2:] = sol2[:,:,depth-n-1:depth-n:] + sol2[:,:,depth+n+1:depth+n+2:]

            fsol_length, fsol_width, fsol_depth = p1.shape
             #the solution will not have length, width, or depth larger than p1 so we set values for the final legth, width, and depth
            sol3_length, sol3_width, sol3_depth = sol3.shape
            add_length = fsol_length - sol3_length
            add_width = fsol_width - sol3_width
            add_depth = fsol_depth - sol3_depth
            p2 = np.pad(sol3, ((0,add_length),(0,add_width),(0,add_depth)), 'constant', constant_values = 0)
            Pf = (p1+p2)
            Pf = Pf.astype(float)
            Pf = .5*Pf
            return MultiCheb(Pf)

        else:
            Mon = np.zeros_like(self)
            Mon[idx] = 1
            Mon = MultiCheb(Mon)
            return Mon*MultiCheb(self)
