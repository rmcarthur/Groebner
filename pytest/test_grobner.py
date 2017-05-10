import numpy as np
import os, sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
sys.path.append("../groebner")
import groebner.maxheap
from groebner.multi_power import MultiPower
from groebner.groebner import Groebner
import pytest

#write more test

def test_s_poly():
    a2 = np.array([[0,0,0,1],[0,-2,0,0],[0,0,0,0],[0,0,0,0]])
    a3 = np.array([[0,1,0,0],[0,0,1,0],[-2,0,0,0],[0,0,0,0]])
    c2 = MultiPower(a2.T)
    c3 = MultiPower(a3.T)
    grob = Groebner([c2,c3])
    s1 = np.round(grob.calc_s(c2,c3).coeff)
    true_s = np.array([[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])
    assert np.allclose(s1.all(),true_s.all())



