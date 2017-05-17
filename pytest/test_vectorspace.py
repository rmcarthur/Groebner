import numpy as np
import os, sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/groebner')
from vector_space import VectorSpace
from multi_power import MultiPower
from multi_cheb import MultiCheb
import pytest
import pdb

def test_makeBasis():
    f1 = MultiPower(np.array([[0,-1.5,.5],[-1.5,1.5,0],[1,0,0]]))
    f2 = MultiPower(np.array([[0,0,0],[-1,0,1],[0,0,0]]))
    f3 = MultiPower(np.array([[0,-1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
    G = [f1, f2, f3]
    vs = VectorSpace()
    basis = vs.makeBasis(G)
    trueBasis = [(0,0), (1,0), (0,1), (1,1), (0,2)]

    assert (len(basis) == len(trueBasis)) and (m in basis for m in trueBasis), \
            "Failed on MultiPower in 2 vars."

    
