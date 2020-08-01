from shapely.geometry import *

def getExtrapoledLine(p1,p2):
    'Creates a line extrapoled in p1->p2 direction'
    EXTRAPOL_RATIO = 10
    a = p1
    b = (p1[0]+EXTRAPOL_RATIO*(p2[0]-p1[0]), p1[1]+EXTRAPOL_RATIO*(p2[1]-p1[1]) )
    return LineString([a,b])

line=LineString([(3,0),(3,5),(7,9.5)])
box=Polygon([(0,0),(0,10),(10,10),(10,0)])

box_ext = LinearRing(box.exterior.coords) #we only care about the boundary intersection
l_coords = list(line.coords)
long_line = getExtrapoledLine(*l_coords[-2:]) #we use the last two points

if box_ext.intersects(long_line):
    intersection_points = box_ext.intersection(long_line)
    new_point_coords = list(intersection_points.coords)[0] #
else:
    raise Exception("Something went really wrong")

l_coords.append(new_point_coords)
new_extended_line = LineString(l_coords) 

# To see the problem:
import pylab
x, y = box.exterior.xy
pylab.plot(x,y)
l_coords = list(line.coords)
x = [p[0] for p in l_coords]
y = [p[1] for p in l_coords]
pylab.plot(x,y)
longl_coords = list(long_line.coords)
x = [p[0] for p in longl_coords]
y = [p[1] for p in longl_coords]
pylab.plot(x,y)
pylab.plot(new_point_coords[0], new_point_coords[1], 'o')
pylab.show()

# To see the solution:
x, y = box.exterior.xy
pylab.plot(x,y)
l_coords = list(new_extended_line.coords)
x = [p[0] for p in l_coords]
y = [p[1] for p in l_coords]
pylab.plot(x,y)
pylab.show()
