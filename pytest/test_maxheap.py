import numpy as np
import os, sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/groebner')
from maxheap import MaxHeap
import pytest



def test_push_pop():
    a0 = (0,0,1,0,0)
    a1 = (0,1,1,3,1)
    a2 = (0,1,1,3,0,0,0,1)
    a3 = (2,2,2,3,4,1,4,3)
    a4 = (0,1,1,2,2)
    maxh = MaxHeap()
    maxh.heappush(a1)
    maxh.heappush(a3)
    maxh.heappush(a0)
    maxh.heappush(a2)
    maxh.heappush(a4)
    assert maxh.heappop() == a3
    assert maxh.heappop() == a1
    
    maxh.heappush(a3)
    maxh.heappush(a3)
    
    assert maxh.heappop() == a3
    assert maxh.heappop() == a2
    assert maxh.heappop() == a4
    assert maxh.heappop() == a0




