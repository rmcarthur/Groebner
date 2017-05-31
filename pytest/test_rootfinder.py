import numpy as np
import os, sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/groebner')
from root_finder import RootFinder
from multi_power import MultiPower
from multi_cheb import MultiCheb
import pytest
import pdb

def test_makeBasis():
    f1 = MultiPower(np.array([[0,-1.5,.5],[-1.5,1.5,0],[1,0,0]]))
    f2 = MultiPower(np.array([[0,0,0],[-1,0,1],[0,0,0]]))
    f3 = MultiPower(np.array([[0,-1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
    G = [f1, f2, f3]
    rf = RootFinder(G)
    basis = rf.vectorBasis
    trueBasis = [(0,0), (1,0), (0,1), (1,1), (0,2)]

    assert (len(basis) == len(trueBasis)) and (m in basis for m in trueBasis), \
            "Failed on MultiPower in 2 vars."

    f1 = MultiPower(np.array([[[0,0,1],[0,3/20,0],[0,0,0]],
                              [[0,0,0],[-3/40,1,0],[0,0,0]],
                              [[0,0,0],[0,0,0],[0,0,0]]]))

    f2 = MultiPower(np.array([[[3/16,-5/2,0],[0,3/16,0],[0,0,0]],
                              [[0,0,1],[0,0,0],[0,0,0]],
                              [[0,0,0],[0,0,0],[0,0,0]]]))

    f3 = MultiPower(np.array([[[0,1,1/2],[0,3/40,1],[0,0,0]],
                              [[-1/2,20/3,0],[-3/80,0,0],[0,0,0]],
                              [[0,0,0],[0,0,0],[0,0,0]]]))

    f4 = MultiPower(np.array([[[3/32,-7/5,0,1],[-3/16,83/32,0,0],[0,0,0,0],[0,0,0,0]],
                              [[3/40,-1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                              [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                              [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]]))

    f5 = MultiPower(np.array([[[5,0,0],[0,0,0],[0,0,0]],
                              [[0,-2,0],[0,0,0],[0,0,0]],
                              [[1,0,0],[0,0,0],[0,0,0]]]))

    f6 = MultiPower(np.array([[[0,0,0],[0,0,0],[1,0,0]],
                              [[0,-8/3,0],[0,0,0],[0,0,0]],
                              [[0,0,0],[0,0,0],[0,0,0]]]))

    G = [f1, f2, f3, f4, f5, f6]
    basis = rf.makeVectorBasis(G)
    trueBasis = [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(0,0,2),(1,0,1),(0,1,1)]

    assert (len(basis) == len(trueBasis)) and (m in basis for m in trueBasis), \
            "Failed on MultiPower in 3 vars."

def testCoordinateVector():
    f1 = MultiCheb(np.array([[0,-1.5,.5],[-1.5,1.5,0],[1,0,0]]))
    f2 = MultiCheb(np.array([[0,0,0],[-1,0,1],[0,0,0]]))
    f3 = MultiCheb(np.array([[0,-1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
    G = [f1, f2, f3]
    rf = RootFinder(G)
