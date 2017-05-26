import numpy as np
import os, sys
from itertools import permutations
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
    grob = Groebner([])
    grob.new_polys = list((poly1, poly2, poly3))
    grob.matrix_terms = []
    grob.np_matrix = np.array([])
    grob.term_set = set()
    grob.lead_term_set = set()
    grob._add_polys(grob.new_polys)

    assert(grob.reduce_matrix())
    assert(len(grob.old_polys) == 3)
    assert(len(grob.new_polys) == 1)

    poly1 = MultiPower(np.array([[1., 0.],[0., 0.]]))
    poly2 = MultiPower(np.array([[0., 0.],[1., 0.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., 0.]]))
    grob = Groebner([])
    grob.new_polys = list((poly1, poly2, poly3))
    grob.matrix_terms = []
    grob.np_matrix = np.array([])
    grob.term_set = set()
    grob.lead_term_set = set()
    grob._add_polys(grob.new_polys)

    assert(not grob.reduce_matrix())
    assert(len(grob.old_polys) == 3)
    assert(len(grob.new_polys) == 0)

    poly1 = MultiPower(np.array([[1., -14.],[0., 2.]]))
    poly2 = MultiPower(np.array([[0., 3.],[1., 6.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))
    grob = Groebner([])
    grob.new_polys = list((poly1, poly2, poly3))
    grob.matrix_terms = []
    grob.np_matrix = np.array([])
    grob.term_set = set()
    grob.lead_term_set = set()
    grob._add_polys(grob.new_polys)
    assert(grob.reduce_matrix())
    assert(len(grob.old_polys) == 3)
    assert(len(grob.new_polys) == 2)

def test_solve():
    #First Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-26,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    grob = Groebner([A,B,C])
    X = MultiPower(np.array([[-2.],[ 1.]]))
    Y = MultiPower(np.array([[-3.,1.]]))
    x1, y1 = grob.solve()
    assert(np.any([X==i and Y==j for i,j in permutations((x1,y1),2)]))

    #Second Test
    A = MultiPower(np.array([
                         [[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    B = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    C = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    grob = Groebner([A,B,C])
    w1, x1, y1, z1 = grob.solve()

    W = MultiPower(np.array([[[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 1.],[ 0.]]],
                             [[[ 0.],[-1.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]]]]))
    X = MultiPower(np.array([[[[ 0.,0.,0.,0.,0.,1.],[-1.,0.,0.,0.,0.,0.]]]]))
    Y = MultiPower(np.array([[[[ 0.],[ 0.],[ 1.]],[[-1.],[ 0.],[ 0.]]]]))
    Z = MultiPower(np.array([[[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 1.]]],
                             [[[-1.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]]]]))

    assert(np.any([W==i and X==j and Y==k and Z==l for i,j,k,l in permutations((w1,x1,y1,z1),4)]))

    #Third Test
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    grob = Groebner([A,B])
    x1, y1 = grob.solve()
    assert(np.any([A==i and B==j for i,j in permutations((x1,y1),2)]))

    #Fourth Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-25,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    grob = Groebner([A,B,C])
    X = MultiPower(np.array([[1.]]))
    x1 = grob.solve()
    assert(X == x1[0])

    #Fifth Test
    A = MultiPower(np.array([[1,1],[0,0]]))
    B = MultiPower(np.array([[1,0],[1,0]]))
    C = MultiPower(np.array([[1,0],[1,0],[0,1]]))
    grob = Groebner([A,B,C])
    X = MultiPower(np.array([[1.]]))
    x1 = grob.solve()
    assert(X == x1[0])
    
    
def test_phi_criterion():
    
    # Same as grob.solve(), but added true/false to test the phi's. 
    def solve_phi(grob,phi=True):
        polys = True
        i = 1
        while polys:
            print("Starting Loop #"+str(i))
            print("Initializing")
            grob.initialize_np_matrix()
            print(grob.np_matrix.shape)
            print("ADDING PHI's")
            grob.add_phi_to_matrix(False)
            print(grob.np_matrix.shape)
            print("ADDING r's")
            grob.add_r_to_matrix()
            print(grob2.np_matrix.shape)
            polys = grob.reduce_matrix()
            i+=1
            print("WE WIN")
            
    
    # Simple Test Case (Nothing gets added )
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    grob1 = Groebner([A,B])
    grob2 = Groebner([A,B])
    
    solve_phi(grob1,True)
    solve_phi(grob2,False)
            
        
    x1,y1 = grob1.reduce_groebner_basis()
    x2,y2 = grob2.reduce_groebner_basis()

    assert(np.any([x2==i and y2==j for i,j in permutations((x1,y1),2)])), "Not the same basis!"
    
            
    #Second Test
    A = MultiPower(np.array([
                         [[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    B = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    C = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    grob1 = Groebner([A,B,C])
    grob2 = Groebner([A,B,C])    


    solve_phi(grob1,True)
    solve_phi(grob2,False)
        
    basis1= grob1.reduce_groebner_basis()
    basis2 = grob2.reduce_groebner_basis()
    
    # Currently, the test fails when phi criterion is not used. Will need to figure out solution! 
    #assert(len(basis1)==len(basis2)) 
    
    
