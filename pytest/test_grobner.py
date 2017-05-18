import numpy as np
import os, sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/groebner')
import maxheap
from multi_power import MultiPower
from groebner_class import Groebner
import pytest

#write more tests

def test_reduce_matrix():
    poly1 = MultiPower(np.array([[1., 0.],[0., 1.]]))
    poly2 = MultiPower(np.array([[0., 0.],[1., 0.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))
    grob = Groebner([poly1, poly2, poly3])
    grob.initialize_np_matrix()
    assert(grob.reduce_matrix())
    assert(len(grob.old_polys) == 3)
    assert(len(grob.new_polys) == 1)
    
    poly1 = MultiPower(np.array([[1., 0.],[0., 0.]]))
    poly2 = MultiPower(np.array([[0., 0.],[1., 0.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., 0.]]))
    grob = Groebner([poly1, poly2, poly3])
    grob.initialize_np_matrix()
    assert(not grob.reduce_matrix())
    assert(len(grob.old_polys) == 3)
    assert(len(grob.new_polys) == 0)
    
    poly1 = MultiPower(np.array([[1., -14.],[0., 2.]]))
    poly2 = MultiPower(np.array([[0., 3.],[1., 6.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))
    grob = Groebner([poly1, poly2, poly3])
    grob.initialize_np_matrix()
    assert(grob.reduce_matrix())
    assert(len(grob.old_polys) == 3)
    assert(len(grob.new_polys) == 2)    
    