import numpy as np
import os, sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from groebner.multi_cheb import MultiCheb
import pytest
import pdb

def test_add():
    a1 = np.arange(27).reshape((3,3,3))
    Test2 = MultiCheb(a1)
    a2 = np.ones((3,3,3))
    Test3 = MultiCheb(a2)
    addTest = Test2 + Test3
    
    assert addTest.coeff.all() == (Test2.coeff + Test3.coeff).all()
    
def test_mult():
    test1 = np.array([[0,1],[2,1]])
    test2 = np.array([[2,2],[3,0]])
    cheb1 = MultiCheb(test1)
    cheb2 = MultiCheb(test2)
    new_cheb = cheb1*cheb2
    truth = np.array([[4, 3.5, 1],[5,9,1],[3,1.5,0]]) 
    
    assert np.allclose(new_cheb.coeff.all() ,truth.all())

def test_mult_diff():
    '''
    #TODO: Verify by hand...
    Test implementation with different shape sizes
    '''
    c1 = MultiCheb(np.arange(0,4).reshape(2,2))
    c2 = MultiCheb(np.ones((2,1)))
    p = c1*c2
    truth = np.array([[1,2.5,0],[2,4,0],[1,1.5,0]])
    
    assert np.allclose(p.coeff.all(),truth.all())



