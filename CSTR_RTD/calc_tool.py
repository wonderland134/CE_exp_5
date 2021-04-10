import numpy as np

class calc_tool():
    def __init__(self):
        self.info = None

    def round_sig(x, sig):
        a = np.log10(abs(x))
        if a >= 0:
            n = np.trunc(a)
        elif a < 0:
            n = np.trunc(a)-1
        return np.round_(x, int(-1*(n+1-sig)))
    
    def get_angle_by_points(p1, p2, p3):
        #p1, p2, p3가 이루는 각의 크기를 반환함(호도법)
        p1x, p1y = p1
        p2x, p2y = p2
        p3x, p3y = p3
        
        if p1x == p2x and p1y != p2y:
            m1 = 'inf'
        elif p1y == p2y and p1x == p2x:
            return 0
        elif p1y == p2y:
            m1 = 0
        else:
            m1 = (p2y-p1y)/(p2x-p1x)       #slope between p1 - p2
        
        if p2x == p3x and p2y != p3y:
            m2 = 'inf'
        elif p2y == p3y and p2x == p3x:
            return 0
        elif p2y == p3y:
            m2 = 0
        else:
            m2 = (p3y-p2y)/(p3x-p2x)       #slope between p1 - p2
        
        if m1 == 'inf' and m2 == 0:
            theta = np.pi/2
        elif m1 == 0 and m2 == 'inf':
            theta = np.pi/2
        elif m1 == 'inf' and m2 == 'inf':
            theta = 0
        elif m1 == 0 and m2 == 0:
            theta = 0
        elif m1 == 'inf' and m2 != 0 and m2 != 'inf':
            tan_theta = abs(-1/m2)
            theta = abs(np.arctan(tan_theta))
        elif m2 == 'inf' and m1 != 0 and m1 != 'inf':
            tan_theta = abs(1/m1)
            theta = abs(np.arctan(tan_theta))
        elif m1*m2 == -1:
            theta = np.pi/2
        else:
            tan_theta = abs((m2-m1)/(1+m2*m1))
            theta = abs(np.arctan(tan_theta))
        
        return theta
    
    def distance_two_points(p1, p2):
        #p1 p2사이의 거리
        p1x, p1y = p1
        p2x, p2y = p2
        
        return ((p1x-p2x)**2+(p1y-p2y)**2)**0.5
    
    def distance_point_to_line_3points(point, points_config_line):
        x0, y0 = point
        p1, p2 = points_config_line
        p1x, p1y = p1
        p2x, p2y = p2
        
        if p1x == p2x:
            dist = abs(x0-p1x)
        else:
            slope = (p1y-p2y)/(p1x-p2x)
            #eq : slope x - y +p1y -slope p1x
            dist = abs(slope*x0 + y0*(-1) + p1y - slope*p1x)/(slope**2 + (-1)**2)**0.5

        return dist
    
    def distance_point_to_line(p0, p1, slope):
        x0, y0 = p0
        x1, y1 = p1
        
        if slope == 'inf':
            dist = abs(x1-x0)
        else:
            dist = abs(slope * x0 + (-1) * y0 + y1 - x1 * slope)/(slope**2 + (-1)**2)**0.5
        
        return dist
    
    def make_vec_by_two_points(p1, p2, size = 1):
        p1x, p1y = p1
        p2x, p2y = p2
        v = np.array([p2x-p1x, p2y-p1y])
        norm = (v[0]**2+v[1]**2)**0.5
        v = size*v/norm
        return tuple(v)
    
    def get_orthogonal_vec(v0, size = 1):
        vx, vy = v0
        if vx == 0:
            a = size
            b = 0
        else:
            a = (-vy/vx * (vy**2/vx**2+1)**(-0.5))*size
            b = ((vy**2/vx**2+1)**(-0.5))*size

        return (a, b)

if __name__ == '__main__':
    pass