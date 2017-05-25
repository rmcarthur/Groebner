import numpy as np
import random
import os,sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1]) + '/groebner')
from multi_power import MultiPower
import pytest


def test_add():
    a1 = np.arange(27).reshape((3,3,3))
    Test2 = MultiPower(a1)
    a2 = np.ones((3,3,3))
    Test3 = MultiPower(a2)
    addTest = Test2 + Test3
    assert addTest.coeff.all() == (Test2.coeff + Test3.coeff).all()

def test_mult():
    test1 = np.array([[0,1],[2,1]])
    test2 = np.array([[2,2],[3,0]])
    p1 = MultiPower(test1)
    p2 = MultiPower(test2)
    new_poly = p1*p2
    truth = np.array([[0, 2, 2],[4,9,2],[6,3,0]])
    assert np.allclose(new_poly.coeff, truth)



def test_generator():
    poly = MultiPower(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]))
    gen = poly.degrevlex_gen()
    i = 0
    for idx in gen:
        i += 1
        if(i == 1):
            assert (idx == [4., 4.]).all()
        elif(i == 2):
            assert (idx == [4., 3.]).all()
        elif(i == 3):
        	assert (idx == [3., 4.]).all()
        elif(i == 4):
            assert (idx == [4., 2.]).all()
        elif(i == 5):
            assert (idx == [3., 3.]).all()
        elif(i == 6):
            assert (idx == [2., 4.]).all()
        elif(i == 7):
            assert (idx == [4., 1.]).all()
        elif(i == 8):
            assert (idx == [3., 2.]).all()
        elif(i == 9):
            assert (idx == [2., 3.]).all()
        elif(i == 10):
            assert (idx == [1., 4.]).all()
        elif(i == 11):
            assert (idx == [4., 0.]).all()
        elif(i == 12):
            assert (idx == [3., 1.]).all()
        elif(i == 13):
            assert (idx == [2., 2.]).all()
        elif(i == 14):
        	assert (idx == [1., 3.]).all()
        elif(i == 15):
            assert (idx == [0., 4.]).all()
        elif(i == 16):
            assert (idx == [3., 0.]).all()
        elif(i == 17):
            assert (idx == [2., 1.]).all()
        elif(i == 18):
            assert (idx == [1., 2.]).all()
        elif(i == 19):
            assert (idx == [0., 3.]).all()
        elif(i == 20):
            assert (idx == [2., 0.]).all()
        elif(i == 21):
            assert (idx == [1., 1.]).all()
        elif(i == 22):
            assert (idx == [0., 2.]).all()
        elif(i == 23):
            assert (idx == [1., 0.]).all()
        elif(i == 24):
            assert (idx == [0., 1.]).all()
        elif(i == 25):
            assert (idx == [0., 0.]).all()


    poly = MultiPower(np.array([[[[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.]]], [[[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.]]]]),
                      lead_term = (0,0,0,0))
    gen = poly.degrevlex_gen()
    i = 0
    for idx in gen:
        i += 1
        if(i == 1):
        	assert (idx == [ 1.,  1.,  1.,  2.]).all()
        elif(i == 2):
            assert (idx == [ 1.,  1.,  1.,  1.]).all()
        elif(i == 3):
            assert (idx == [ 1.,  1.,  0.,  2.]).all()
        elif(i == 4):
            assert (idx == [ 1.,  0.,  1.,  2.]).all()
        elif(i == 5):
            assert (idx == [ 0.,  1.,  1.,  2.]).all()
        elif(i == 6):
            assert (idx == [ 1.,  1.,  1.,  0.]).all()
        elif(i == 7):
            assert (idx == [ 1.,  1.,  0.,  1.]).all()
        elif(i == 8):
            assert (idx == [ 1.,  0.,  1.,  1.]).all()
        elif(i == 9):
             assert (idx == [ 0.,  1.,  1.,  1.]).all()
        elif(i == 10):
            assert (idx == [ 1.,  0.,  0.,  2.]).all()
        elif(i == 11):
            assert (idx == [ 0.,  1.,  0.,  2.]).all()
        elif(i == 12):
            assert (idx == [ 0.,  0.,  1.,  2.]).all()
        elif(i == 13):
            assert (idx == [ 1.,  1.,  0.,  0.]).all()
        elif(i == 14):
            assert (idx == [ 1.,  0.,  1.,  0.]).all()
        elif(i == 15):
            assert (idx == [ 0.,  1.,  1.,  0.]).all()
        elif(i == 16):
            assert (idx == [ 1.,  0.,  0.,  1.]).all()
        elif(i == 17):
            assert (idx == [ 0.,  1.,  0.,  1.]).all()
        elif(i == 18):
            assert (idx == [ 0.,  0.,  1.,  1.]).all()
        elif(i == 19):
            assert (idx == [ 0.,  0.,  0.,  2.]).all()
        elif(i == 20):
            assert (idx == [ 1.,  0.,  0.,  0.]).all()
        elif(i == 21):
            assert (idx == [ 0.,  1.,  0.,  0.]).all()
        elif(i == 22):
            assert (idx == [ 0.,  0.,  1.,  0.]).all()
        elif(i == 23):
            assert (idx == [ 0.,  0.,  0.,  1.]).all()
        elif(i == 24):
            assert (idx == [ 0.,  0.,  0.,  0.]).all()

'''
def test_mon_mult():
    dim_options = [2,3,5,8,15,16,41]
    dim1 = dim_options[random.randint(1,7)]
    dim2 = dim_options[random.randint(1,7)]
    polynomial = np.random.randint(100, size = (dim1,dim2))
    for i in range(20):
        for j in range(20):
            mon = (i,j)
            '''
