import multi_cheb
from multi_cheb import MultiCheb
import numpy as np
   
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
    add_a_list = np.zeros((len(new_shape),2))
    add_b_list = np.zeros((len(new_shape),2))
    add_a_list[:,1] = add_a
    add_b_list[:,1] = add_b
    a = MultiCheb(np.pad(a.coeff,add_a_list.astype(int),'constant'), clean_zeros = False)
    b = MultiCheb(np.pad(b.coeff,add_b_list.astype(int),'constant'), clean_zeros = False)
    return a,b

def fold_for_reg_mult(temp, half, dim_to_fold, dim):
    slice0 = slice(None, half+1, None)
    slice1 = slice(None, None, -1)
    slice2 = slice(half,None,None)
    slice3 = slice(0,1,None)
    indexer0 = [slice(None,None,None)]*dim
    indexer1 = [slice(None,None,None)]*dim
    indexer2 = [slice(None,None,None)]*dim

    indexer0[dim_to_fold] = slice0
    indexer1[dim_to_fold] = slice1
    indexer2[dim_to_fold] = slice2

    p2 = temp[indexer0][indexer1] + temp[indexer2]
    indexer2[dim_to_fold] = slice3
    p2[indexer2] = p2[indexer2]/2.

    return p2



def mult(self,other):
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
    shape_of_temp = temp.shape
    dim = temp.ndim
    for i in range(dim):
        half = shape_of_temp[i]//2
        p2 = fold_for_reg_mult(p2, half, i, dim)

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
    #TODO: You can use the lead_term kwarg to save some time



if __name__ == "__main__":
    A = np.arange(1,10).reshape(3,3)
    B = np.arange(1,17).reshape(4,4)
    C = np.arange(1,26).reshape(5,5)
    D = np.arange(1,13).reshape(3,4)
    M = np.zeros((4,3,2))
    M[0,1,1] = 1
    print(mult(A,M))
