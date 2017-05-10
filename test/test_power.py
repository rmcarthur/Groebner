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

    def test_generator(self):
        poly = MultiPower(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]))
        gen = poly.degrevlex_gen()
        i = 0
        for idx in gen:
            i += 1
            if(i == 1):
                self.assertTrue((idx == [4., 4.]).all())
            elif(i == 2):
                self.assertTrue((idx == [4., 3.]).all())
            elif(i == 3):
                self.assertTrue((idx == [3., 4.]).all())
            elif(i == 4):
                self.assertTrue((idx == [4., 2.]).all())
            elif(i == 5):
                self.assertTrue((idx == [3., 3.]).all())
            elif(i == 6):
                self.assertTrue((idx == [2., 4.]).all())
            elif(i == 7):
                self.assertTrue((idx == [4., 1.]).all())
            elif(i == 8):
                self.assertTrue((idx == [3., 2.]).all())
            elif(i == 9):
                self.assertTrue((idx == [2., 3.]).all())
            elif(i == 10):
                self.assertTrue((idx == [1., 4.]).all())
            elif(i == 11):
                self.assertTrue((idx == [4., 0.]).all())
            elif(i == 12):
                self.assertTrue((idx == [3., 1.]).all())
            elif(i == 13):
                self.assertTrue((idx == [2., 2.]).all())
            elif(i == 14):
                self.assertTrue((idx == [1., 3.]).all())
            elif(i == 15):
                self.assertTrue((idx == [0., 4.]).all())
            elif(i == 16):
                self.assertTrue((idx == [3., 0.]).all())
            elif(i == 17):
                self.assertTrue((idx == [2., 1.]).all())
            elif(i == 18):
                self.assertTrue((idx == [1., 2.]).all())
            elif(i == 19):
                self.assertTrue((idx == [0., 3.]).all())
            elif(i == 20):
                self.assertTrue((idx == [2., 0.]).all())
            elif(i == 21):
                self.assertTrue((idx == [1., 1.]).all())
            elif(i == 22):
                self.assertTrue((idx == [0., 2.]).all())
            elif(i == 23):
                self.assertTrue((idx == [1., 0.]).all())
            elif(i == 24):
                self.assertTrue((idx == [0., 1.]).all())
            elif(i == 25):
                self.assertTrue((idx == [0., 0.]).all())


        poly = MultiPower(np.array([[[[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.]]], [[[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.]]]]), lead_term = (0,0,0,0))
        gen = poly.degrevlex_gen()
        i = 0
        for idx in gen:
            i += 1
            if(i == 1):
                self.assertTrue((idx == [ 1.,  1.,  1.,  2.]).all())
            elif(i == 2):
                self.assertTrue((idx == [ 1.,  1.,  1.,  1.]).all())
            elif(i == 3):
                self.assertTrue((idx == [ 1.,  1.,  0.,  2.]).all())
            elif(i == 4):
                self.assertTrue((idx == [ 1.,  0.,  1.,  2.]).all())
            elif(i == 5):
                self.assertTrue((idx == [ 0.,  1.,  1.,  2.]).all())
            elif(i == 6):
                self.assertTrue((idx == [ 1.,  1.,  1.,  0.]).all())
            elif(i == 7):
                self.assertTrue((idx == [ 1.,  1.,  0.,  1.]).all())
            elif(i == 8):
                self.assertTrue((idx == [ 1.,  0.,  1.,  1.]).all())
            elif(i == 9):
                self.assertTrue((idx == [ 0.,  1.,  1.,  1.]).all())
            elif(i == 10):
                self.assertTrue((idx == [ 1.,  0.,  0.,  2.]).all())
            elif(i == 11):
                self.assertTrue((idx == [ 0.,  1.,  0.,  2.]).all())
            elif(i == 12):
                self.assertTrue((idx == [ 0.,  0.,  1.,  2.]).all())
            elif(i == 13):
                self.assertTrue((idx == [ 1.,  1.,  0.,  0.]).all())
            elif(i == 14):
                self.assertTrue((idx == [ 1.,  0.,  1.,  0.]).all())
            elif(i == 15):
                self.assertTrue((idx == [ 0.,  1.,  1.,  0.]).all())
            elif(i == 16):
                self.assertTrue((idx == [ 1.,  0.,  0.,  1.]).all())
            elif(i == 17):
                self.assertTrue((idx == [ 0.,  1.,  0.,  1.]).all())
            elif(i == 18):
                self.assertTrue((idx == [ 0.,  0.,  1.,  1.]).all())
            elif(i == 19):
                self.assertTrue((idx == [ 0.,  0.,  0.,  2.]).all())
            elif(i == 20):
                self.assertTrue((idx == [ 1.,  0.,  0.,  0.]).all())
            elif(i == 21):
                self.assertTrue((idx == [ 0.,  1.,  0.,  0.]).all())
            elif(i == 22):
                self.assertTrue((idx == [ 0.,  0.,  1.,  0.]).all())
            elif(i == 23):
                self.assertTrue((idx == [ 0.,  0.,  0.,  1.]).all())
            elif(i == 24):
                self.assertTrue((idx == [ 0.,  0.,  0.,  0.]).all())

if __name__ == '__main__':
    unittest.main()




