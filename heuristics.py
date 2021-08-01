import math

class Heuristic(object):
    @staticmethod 
    def l2_norm(a, b):
        x1,y1 = a
        x2,y2 = b
        return math.sqrt((x2-x1)**2+(y2-y1)**2)
    @staticmethod
    def l1_norm(a,b):
        x1,y1 = a
        x2,y2 = b
        return abs(x1-x2)+abs(y1-y2)
    @staticmethod
    def grid_dist(a,b):
        x1,y1 = a
        x2,y2 = b
        dx = abs(x2-x1)
        dy = abs(y2-y1)
        low = min(dx,dy)
        dd = abs(dy-dx)
        return low*1.414 + dd