import numpy as np
import groebner_class, multi_power, multi_cheb
from multi_cheb import MultiCheb


def mon_mult(self, idx):
    '''Function creates a matrix of zeroes of the correct dimmension
    and then replaces the values in the matrix with the values given
    after adding corresponding terms

    This function works in 2D only

    To use this function pass the monomial in as a tuple corresponding to
    the values of the Chebychev polynomial. For example T_2(x)T_3(y) would
    be (2,3). This is idx and P is the polynomial in matrix form.
    '''
    pad_values = list()
    for i in idx: #iterates through monomial and creates a tuple of pad values for each dimension
        pad_dim_i = (i,0)
        #In np.pad each dimension is a tuple of (i,j) where i is how many to pad in front and j is how many to pad after.
        pad_values.append(pad_dim_i)
    p1 = np.pad(self, (pad_values), 'constant', constant_values = 0)

    row,col,depth  = idx
    num_rows,num_col,num_z = self.shape
    sol = np.zeros_like(self)
    sol[:1:,:,:] = self[row:row+1:,:,:]
    for n in range(num_rows):
        if row-n-1 < 0:
            if row+n+2 > num_rows:
                break
            else:
                sol[n+1:n+2:,:,:] = self[row+n+1:row+n+2:,:,:]
        else:
            if row+n+2 > num_rows:
                sol[n+1:n+2:,:,:] = self[row-n-1:row-n:,:,:]
            else:
                sol[n+1:n+2:,:,:] = self[row-n-1:row-n:,:,:] + self[row+n+1:row+n+2:,:,:]
    sol2 = np.zeros(sol_idx2)
    sol2 = np.zeros_like(self)
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
    sol3 = np.zeros_like(self, dtype = np.float)
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
    sol3_length, sol3_width, sol3_depth = sol3.shape
    add_length = fsol_length - sol3_length
    add_width = fsol_width - sol3_width
    add_depth = fsol_depth - sol3_depth
    p2 = np.pad(sol3, ((0,add_length),(0,add_width),(0,add_depth)), 'constant', constant_values = 0)
    Pf = (p1+p2)
    Pf = .5*Pf
    return Pf

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


def mon_mult2(self,idx):
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
        solution_matrix = fold_in_i_dir(solution_matrix, number_of_dim, i, shape_of_self[i], idx[i])
    #return solution_matrix
    big = p1.shape
    little = solution_matrix.shape
    pad_list = list()

    for i,j in zip(big,little):
        z = i-j
        pad_list.append((0,z))

    p2 = np.pad(solution_matrix, (pad_list), 'constant', constant_values = 0)
    Pf = (p1+p2)
    Pf = .5*Pf
    return Pf


possible_dim = np.random.random_integers(1,25,(100))
dim_number = np.random.choice(possible_dim, (random.randint(1,8)))
Poly3 = MultiCheb(np.random.randint(0,100,(dim_number)))



if __name__ == "__main__":
    N = np.arange(1,28).reshape(3,3,3)
    M1 = np.arange(1,9).reshape(2,2,2)
    #M = M1/4.0
    #print(mon_mult(M, (0,1,1)))
    print(mon_mult2(M1, (0,1,1)))
    print(mon_mult(N, (2,1,2)))
