import numpy as np
import os, sys
if (os.name == 'nt'):
    sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1]) + '/groebner')
else:
    sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/groebner')
from multi_cheb import MultiCheb
from multi_power import MultiPower
from convert_poly import cheb_poly, poly_cheb
import pytest
import pdb
import random

def test_cheb_poly():
    c1 = MultiCheb(np.array([[0,0,0],[0,0,0],[0,1,1]]))
    c2 = cheb_poly(c1)
    truth = np.array([[1,-1,-2],[0,0,0],[-2,2,4]])
    assert np.allclose(truth,c2.coeff)

def test_poly_cheb():
    P = MultiPower(np.array([[1,-1,-2],[0,0,0],[-2,2,4]]))
    c_new = poly_cheb(P)
    truth = np.array(np.array([[0,0,0],[0,0,0],[0,1,1]]))

    assert np.allclose(truth, c_new.coeff)
