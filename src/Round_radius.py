# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:32:21 2019

@author: Beta
"""


import math
 
def getCircle(x1, y1, x2, y2, x3, y3):
    x21 = x2 - x1
    y21 = y2 - y1
    x32 = x3 - x2
    y32 = y3 - y2
    # three colinear
    xy21 = x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1
    xy32 = x3 * x3 - x2 * x2 + y3 * y3 - y2 * y2
    y0 = (x32 * xy21 - x21 * xy32) / (2 * (y21 * x32 - y32 * x21) + 0.01)
    x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21 + 0.01)
    R = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return int(x0), int(y0), int(R)

def Inornot(x0,y0,x,y,Radius):
    if x0 >= x - Radius and x0 <= x + Radius and y0 >= y - Radius and y0 <= y + Radius:
        return True 
    else:
        return False

def Distance(x,y,x0,y0):
    dis = math.sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0))
    return dis
    