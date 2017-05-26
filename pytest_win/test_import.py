import numpy as np
import pytest
import sys, os
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1]) + '/groebner')
import maxheap
from multi_cheb import MultiCheb
from multi_power import MultiPower
from polynomial import Polynomial
from groebner_class import Groebner
