import numpy as np
import os, sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/groebner')
from root_finder import RootFinder
from multi_power import MultiPower
from multi_cheb import MultiCheb
from groebner_class import Groebner
import pytest
import pdb

def test_makeVectorBasis():
    f1 = MultiPower(np.array([[0,-1.5,.5],[-1.5,1.5,0],[1,0,0]]))
    f2 = MultiPower(np.array([[0,0,0],[-1,0,1],[0,0,0]]))
    f3 = MultiPower(np.array([[0,-1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
    G = [f1, f2, f3]
    rf = RootFinder(G)
    basis = rf.vectorBasis
    trueBasis = [(0,0), (1,0), (0,1), (1,1), (0,2)]

    assert (len(basis) == len(trueBasis)) and (m in basis for m in trueBasis), \
            "Failed on MultiPower in 2 vars."

def test_makeVectorBasis_2():
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
    rf = RootFinder(G)
    basis = rf.makeVectorBasis(G)
    trueBasis = [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(0,0,2),(1,0,1),(0,1,1)]

    assert (len(basis) == len(trueBasis)) and (m in basis for m in trueBasis), \
            "Failed on MultiPower in 3 vars."

def testMultMatrix():
    f1 = MultiPower(np.array([[[5,0,0],[0,0,0],[0,0,0]],
                          [[0,-2,0],[0,0,0],[0,0,0]],
                          [[1,0,0],[0,0,0],[0,0,0]]]))

    f2 = MultiPower(np.array([[[1,0,0],[0,1,0],[0,0,0]],
                          [[0,0,0],[0,0,0],[1,0,0]],
                          [[0,0,0],[0,0,0],[0,0,0]]]))

    f3 = MultiPower(np.array([[[0,0,0],[0,0,0],[3,0,0]],
                          [[0,-8,0],[0,0,0],[0,0,0]],
                          [[0,0,0],[0,0,0],[0,0,0]]]))

    F = [f1, f2, f3]
    Gr = Groebner(F)
    rf = RootFinder(Gr)

    x = MultiPower(np.array([[0],[1]]))
    y = MultiPower(np.array([[0,1]]))
    z = MultiPower(np.array([[[0,1]]]))

    mx_RealEig = [eig.real for eig in \
        np.linalg.eigvals(rf.multMatrix(x)) if (eig.imag == 0)]

    my_RealEig = [eig.real for eig in \
        np.linalg.eigvals(rf.multMatrix(y)) if (eig.imag==0)]

    mz_RealEig = [eig.real for eig in \
        np.linalg.eigvals(rf.multMatrix(z)) if (eig.imag==0)]

    assert(len(mx_RealEig) == 2)
    assert(len(my_RealEig) == 2)
    assert(len(mz_RealEig) == 2)
    assert(np.allclose(mx_RealEig, [-1.100987715, .9657124563], atol=1.e-8))
    assert(np.allclose(my_RealEig, [-2.878002536, -2.81249605], atol=1.e-8))
    assert(np.allclose(mz_RealEig, [3.071618528, -2.821182227], atol=1.e-8))

def testMultMatrix_2():
    f1 = MultiPower(np.array([[0,-1.5,.5],[-1.5,1.5,0],[1,0,0]]))
    f2 = MultiPower(np.array([[0,0,0],[-1,0,1],[0,0,0]]))
    f3 = MultiPower(np.array([[0,-1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
    G = [f1, f2, f3]
    rf = RootFinder(G)

    x = MultiPower(np.array([[0],[1]]))
    y = MultiPower(np.array([[0,1]]))

    mx_Eig = np.linalg.eigvals(rf.multMatrix(x))
    my_Eig = np.linalg.eigvals(rf.multMatrix(y))

    assert(len(mx_Eig) == 5)
    assert(len(my_Eig) == 5)
    assert(np.allclose(mx_Eig, [-1., 2., 1., 1., 0.]))
    assert(np.allclose(my_Eig, [1., -1., 1., -1., 0.]))
