import numpy as np
import os, sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/groebner')
import maxheap
from multi_power import MultiPower
from groebner_class import Groebner
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

def test_reduce_matrix():
    poly1 = MultiPower(np.array([[1., 0.],[0., 1.]]))
    poly2 = MultiPower(np.array([[0., 0.],[1., 0.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))
    grob = Groebner([poly1, poly2, poly3])
    grob.polys = [poly1, poly2, poly3]
    assert(grob.reduce_matrix())
    assert(len(grob.old_polys) == 3)
    #assert(len(groebner.new_polys) == 1)
    
    poly1 = MultiPower(np.array([[1., 0.],[0., 0.]]))
    poly2 = MultiPower(np.array([[0., 0.],[1., 0.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))
    groebner = Groebner([poly1, poly2, poly3])
    groebner.polys = [poly1, poly2, poly3]
    assert(not groebner.reduce_matrix())
    assert(len(groebner.old_polys) == 3)
    assert(len(groebner.new_polys) == 0)
    
    poly1 = MultiPower(np.array([[1., -14.],[0., 2.]]))
    poly2 = MultiPower(np.array([[0., 3.],[1., 6.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))
    groebner = Groebner([poly1, poly2, poly3])
    groebner.polys = [poly1, poly2, poly3]
    assert(groebner.reduce_matrix())
    assert(len(groebner.old_polys) == 3)
    #assert(len(groebner.new_polys) == 2)    
    
    
if __name__ == '__main__': 
    a2 = np.array([[0,0,0,1],[0,-2,0,0],[0,0,0,0],[0,0,0,0]])
    a3 = np.array([[0,1,0,0],[0,0,1,0],[-2,0,0,0],[0,0,0,0]])
    c2 = MultiPower(a2.T)
    c3 = MultiPower(a3.T)
    print(type(c2) == MultiPower)
    print(type(c3) == MultiPower)
    p_list = [c2, c3]
    
    if all([type(p) == MultiPower for p in p_list]):
        power = True
    elif all([type(p) == MultiCheb for p in p_list]):
        power = False
    else:
        print([type(p) == MultiCheb for p in p_list])
        raise ValueError('Bad polynomials in list')
    print(power)


    g = Groebner([c2,c3])

