from matplotlib import pyplot
from shapely.geometry import Point
from descartes import PolygonPatch
import numpy as np
from contour import bacteria_polygon

BLUE = '#6699cc'
GRAY = '#999999'
GM = (np.sqrt(5)-1.0)/2.0
W = 8.0
H = W*GM
SIZE = (W, H)

def set_limits(ax, x0, xN, y0, yN):
    ax.set_xlim(x0, xN)
    ax.set_xticks(range(x0, xN+1))
    ax.set_ylim(y0, yN)
    ax.set_yticks(range(y0, yN+1))
    ax.set_aspect("equal")

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

a = bacteria_polygon([0.5, 4.3, 15, 5.3, 0, 0], 11).polygon
a_boundary= np.array(list(map(list, list(a.exterior.coords))))
b = bacteria_polygon([0.56, 3.4, 12, -0.3, 2, 4], 11).polygon
b_boundary= np.array(list(map(list, list(b.exterior.coords))))


#2
ax = fig.add_subplot(111)

patch1 = PolygonPatch(a, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
ax.add_patch(patch1)
ax.plot(a_boundary[:, 0], a_boundary[:, 1], color='g')
patch2 = PolygonPatch(b, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
ax.add_patch(patch2)
ax.plot(b_boundary[:, 0], b_boundary[:, 1], color='r')
c = a.symmetric_difference(b)

if c.geom_type == 'Polygon':
    patchc = PolygonPatch(c, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
    ax.add_patch(patchc)
elif c.geom_type == 'MultiPolygon':
    for p in c:
        patchp = PolygonPatch(p, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
        ax.add_patch(patchp)

ax.set_title('a.symmetric_difference(b)')
ax.axis('off')
ax.legend()

set_limits(ax, -50, 50, -15, 15)

pyplot.show()

