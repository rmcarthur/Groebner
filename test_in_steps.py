import numpy as np
import pandas as pd
from scipy.linalg import lu
import random
import time

#Local imports
from multi_cheb import MultiCheb
from multi_power import MultiPower
import maxheap
from groebner_class import Groebner
from convert_poly import cheb2poly, poly2cheb
import groebner_basis
from root_finder import roots

matrix1 = np.random.randint(-10,10,(2,3))
matrix2 = np.random.randint(-30,30,(3,3))
T1 = MultiCheb(matrix1)
T2 = MultiCheb(matrix2)
P1 = cheb2poly(T1)
P2 = cheb2poly(T2)

roots([P1,P2])
roots([T1,T2])
