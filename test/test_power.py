import numpy as np
import os,sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
sys.path.append("../groebner")
from groebner.multi_power import MultiPower
import unittest

class TestMultiPower(unittest.TestCase):

    def test_add(self):
        a1 = np.arange(27).reshape((3,3,3))
        Test2 = MultiPower(a1)
        a2 = np.ones((3,3,3))
        Test3 = MultiPower(a2)
        addTest = Test2 + Test3
        self.assertTrue(addTest.coeff.all() == (Test2.coeff + Test3.coeff).all())

    def test_mult(self):
        test1 = np.array([[0,1],[2,1]])
        test2 = np.array([[2,2],[3,0]])
        p1 = MultiPower(test1)
        p2 = MultiPower(test2)
        new_poly = p1*p2
        truth = np.array([[0, 2, 2],[4,9,2],[6,3,0]])
        test = np.allclose(new_poly.coeff, truth)
        self.assertTrue(test)

    def test_generator(self)
        poly = MultiPower(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]))
        gen = poly.degrevlex_gen()
        i = 0
        for idx in gen:
            i += 1
            if(i == 1):
                assertTrue(idx == [4., 4.])
            elif(i == 2):
                assertTrue(idx == [4., 3.])
            elif(i == 3):
                assertTrue(idx == [3., 4.])
            elif(i == 4):
                assertTrue(idx == [4., 2.])
            elif(i == 5):
                assertTrue(idx == [3., 3.])
            elif(i == 6):
                assertTrue(idx == [2., 4.])
            elif(i == 7):
                assertTrue(idx == [4., 1.])
            elif(i == 8):
                assertTrue(idx == [3., 2.])
            elif(i == 9):
                assertTrue(idx == [2., 3.])
            elif(i == 10):
                assertTrue(idx == [1., 4.])
            elif(i == 11):
                assertTrue(idx == [4., 0.])
            elif(i == 12):
                assertTrue(idx == [3., 1.])
            elif(i == 13):
                assertTrue(idx == [2., 2.])
            elif(i == 14):
                assertTrue(idx == [1., 3.])
            elif(i == 15):
                assertTrue(idx == [0., 4.])
            elif(i == 16):
                assertTrue(idx == [3., 0.])
            elif(i == 17):
                assertTrue(idx == [2., 1.])
            elif(i == 18):
                assertTrue(idx == [1., 2.])
            elif(i == 19):
                assertTrue(idx == [0., 3.])
            elif(i == 20):
                assertTrue(idx == [2., 0.])
            elif(i == 21):
                assertTrue(idx == [1., 1.])
            elif(i == 22):
                assertTrue(idx == [0., 2.])
            elif(i == 23):
                assertTrue(idx == [1., 0.])
            elif(i == 24):
                assertTrue(idx == [0., 1.])
            elif(i == 25):
                assertTrue(idx == [0., 0.])


        poly = MultiPower(np.array([[[[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.]]], [[[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.]]]]), lead_term = (0,0,0,0))
        gen = poly.degrevlex_gen()
        i = 0
        for idx in gen:
            i += 1
            if(i == 1):
                assertTrue(idx == [ 1.,  1.,  1.,  2.])
            elif(i == 2):
                assertTrue(idx == [ 1.,  1.,  1.,  1.])
            elif(i == 3):
                assertTrue(idx == [ 1.,  1.,  0.,  2.])
            elif(i == 4):
                assertTrue(idx == [ 1.,  0.,  1.,  2.])
            elif(i == 5):
                assertTrue(idx == [ 0.,  1.,  1.,  2.])
            elif(i == 6):
                assertTrue(idx == [ 1.,  1.,  1.,  0.])
            elif(i == 7):
                assertTrue(idx == [ 1.,  1.,  0.,  1.])
            elif(i == 8):
                assertTrue(idx == [ 1.,  0.,  1.,  1.])
            elif(i == 9):
                assertTrue(idx == [ 0.,  1.,  1.,  1.])
            elif(i == 10):
                assertTrue(idx == [ 1.,  0.,  0.,  2.])
            elif(i == 11):
                assertTrue(idx == [ 0.,  1.,  0.,  2.])
            elif(i == 12):
                assertTrue(idx == [ 0.,  0.,  1.,  2.])
            elif(i == 13):
                assertTrue(idx == [ 1.,  1.,  0.,  0.])
            elif(i == 14):
                assertTrue(idx == [ 1.,  0.,  1.,  0.])
            elif(i == 15):
                assertTrue(idx == [ 0.,  1.,  1.,  0.])
            elif(i == 16):
                assertTrue(idx == [ 1.,  0.,  0.,  1.])
            elif(i == 17):
                assertTrue(idx == [ 0.,  1.,  0.,  1.])
            elif(i == 18):
                assertTrue(idx == [ 0.,  0.,  1.,  1.])
            elif(i == 19):
                assertTrue(idx == [ 0.,  0.,  0.,  2.])
            elif(i == 20):
                assertTrue(idx == [ 1.,  0.,  0.,  0.])
            elif(i == 21):
                assertTrue(idx == [ 0.,  1.,  0.,  0.])
            elif(i == 22):
                assertTrue(idx == [ 0.,  0.,  1.,  0.])
            elif(i == 23):
                assertTrue(idx == [ 0.,  0.,  0.,  1.])
            elif(i == 24):
                assertTrue(idx == [ 0.,  0.,  0.,  0.])

if __name__ == '__main__':
    unittest.main()




